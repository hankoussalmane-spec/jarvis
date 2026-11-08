"""
Jarvis 15 : Architecture complète de jarvis14.py (HUD Réacteur Arc, Outils, Voix Piper/SAPI,
WakeWord "Hey Jarvis") propulsée par Groq Cloud (Whisper Turbo + Llama/Qwen ultra-rapide).
"""

import io
import json
import os
import queue
import re
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path
from types import SimpleNamespace

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import numpy as np
import openwakeword
import sounddevice as sd
from groq import Groq
from openwakeword.model import Model as WakeModel

from core import config, journal, memoire, personnalite, registre, voix
from core.confirmation_vocale import NON, TOUJOURS, interpreter_confirmation
from core.util import nettoyer_reponse_vocale, sans_accents
from tools.lumieres import allumer_si_nuit, charger_pieces_hue

# ---------------------------------------------------------------- GROQ TURBO
GROQ_API_KEY = "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"
_client_groq = Groq(api_key=GROQ_API_KEY)


class WhisperGroqRapide:
    """Remplace faster-whisper CPU (lent) par Whisper-Large-V3-Turbo sur Groq (~0.2s)
    tout en gardant exactement la même interface .transcribe() que jarvis14.py."""

    def transcribe(self, audio, language="fr", beam_size=5):
        tampon_wav = io.BytesIO()
        if isinstance(audio, (str, Path)):
            with open(audio, "rb") as f:
                tampon_wav.write(f.read())
        else:
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(tampon_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
        tampon_wav.seek(0)
        try:
            texte = _client_groq.audio.transcriptions.create(
                file=("audio.wav", tampon_wav.read()),
                model="whisper-large-v3-turbo",
                language=language,
                response_format="text",
            )
            texte_propre = (texte or "").strip()
            return ([SimpleNamespace(text=texte_propre)] if texte_propre else []), None
        except Exception as e:
            print(f"  [Groq STT] erreur : {e}")
            return [], None


class FournisseurGroq:
    """Remplace Ollama local par le meilleur modèle Groq disponible, avec support
    complet des outils (tools) de Jarvis."""

    def __init__(self):
        self.nom = "Groq"
        self.modele = self._choisir_modele()

    def _choisir_modele(self):
        try:
            dispos = [m.id for m in _client_groq.models.list().data]
        except Exception:
            return "llama-3.1-8b-instant"
        preferences = [
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
            "qwen-qwq-32b",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama-3.1-8b-instant",
        ]
        for p in preferences:
            if p in dispos:
                return p
        for m in dispos:
            if not any(x in m.lower() for x in ["whisper", "guard", "tts", "playai", "vision"]):
                return m
        return dispos[0]

    def disponible(self):
        return True

    def _convertir_outils(self, schemas):
        if not schemas:
            return None
        outils_openai = []
        for s in schemas:
            if "function" in s:
                outils_openai.append(s)
            elif "name" in s:
                outils_openai.append({
                    "type": "function",
                    "function": {
                        "name": s["name"],
                        "description": s.get("description", ""),
                        "parameters": s.get("input_schema", s.get("parameters", {"type": "object", "properties": {}})),
                    },
                })
        return outils_openai or None

    def _convertir_messages(self, systeme, historique):
        msgs = [{"role": "system", "content": systeme}]
        for m in historique:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    texte = " ".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
                    tool_calls = []
                    for b in content:
                        if getattr(b, "type", None) == "tool_use":
                            tool_calls.append({
                                "id": b.id,
                                "type": "function",
                                "function": {
                                    "name": b.name,
                                    "arguments": json.dumps(b.input or {}),
                                },
                            })
                    msg_a = {"role": "assistant", "content": texte or None}
                    if tool_calls:
                        msg_a["tool_calls"] = tool_calls
                    msgs.append(msg_a)
                elif role == "user":
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            c = item.get("content", "")
                            if not isinstance(c, str):
                                c = json.dumps(c, ensure_ascii=False)
                            msgs.append({
                                "role": "tool",
                                "tool_call_id": item.get("tool_use_id", "call_1"),
                                "content": c,
                            })
        return msgs

    def repondre(self, systeme, historique, schemas=None):
        msgs = self._convertir_messages(systeme, historique)
        outils = self._convertir_outils(schemas)
        kwargs = {
            "model": self.modele,
            "messages": msgs,
            "temperature": 0.5,
            "max_tokens": 350,
        }
        if outils:
            kwargs["tools"] = outils
            kwargs["tool_choice"] = "auto"

        try:
            resp = _client_groq.chat.completions.create(**kwargs)
        except Exception:
            # Si le modèle refuse un schéma d'outil complexe, repli immédiat en réponse directe
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            resp = _client_groq.chat.completions.create(**kwargs)

        choix = resp.choices[0].message
        blocs = []
        if choix.content:
            texte = choix.content.strip()
            if "</think>" in texte:
                texte = texte.split("</think>")[-1].strip()
            blocs.append(SimpleNamespace(type="text", text=texte))

        if getattr(choix, "tool_calls", None):
            for tc in choix.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                blocs.append(SimpleNamespace(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                ))
            return SimpleNamespace(stop_reason="tool_use", content=blocs)

        return SimpleNamespace(stop_reason="end_turn", content=blocs)


_FOURNISSEUR_GROQ = FournisseurGroq()

# ---------------------------------------------------------------- reglages

MICRO = config.reglage("audio.micro", None)
HAUT_PARLEUR = config.reglage("audio.haut_parleur", None)


def _haut_parleur():
    return config.reglage("audio.haut_parleur", None)


MODELE_WHISPER = "whisper-large-v3-turbo (Groq)"

TAUX = 16000
BLOC = 1280
CAPTURE_TAUX = TAUX
BLOC_CAPTURE = BLOC

SEUIL_REVEIL = config.reglage("assistant.seuil_reveil", 0.5)
SEUIL_INTERRUPTION = config.reglage("assistant.seuil_interruption", 0.7)
SEUIL_PAROLE_SUR = config.reglage("assistant.seuil_parole", 0.025)
SEUIL_SILENCE = config.reglage("assistant.seuil_silence", 0.010)
SILENCE_FIN = config.reglage("assistant.silence_fin", 1.2)
DUREE_MAX = config.reglage("assistant.duree_max", 20)
BLOCS_AVANT_VERIF = 5
DELAI_ENTRE_VERIFS = 1.0
DUREE_SUITE = config.reglage("assistant.duree_suite", 10)

LOG = journal.obtenir()
SENTINEL_CONFIRM = "\x00confirmation\x00"

SYSTEME_BASE = (
    "Tes reponses sont lues a voix haute : reponds en une a deux phrases maximum "
    "(une seule si possible), sans listes, sans titres, sans asterisques ni emoji. "
    "Parle naturellement, en francais. Va a l'essentiel. Ne pose jamais deux fois "
    "la meme question et ne redemande pas une confirmation deja demandee. "
    "Ne commence jamais une reponse finale par une phrase d'attente comme "
    "'attends', 'un instant', 'je regarde' ou 'je cherche'. Le systeme annonce "
    "lui-meme une progression uniquement lorsqu'un outil lent est vraiment lance. "
    "Tu disposes d'outils pour agir sur l'ordinateur : utilise-les quand "
    "l'utilisateur demande une action, et confirme brievement ce que tu as fait."
)

SYSTEME_COURANT = SYSTEME_BASE


def _refaire_systeme(memoire_courante):
    global SYSTEME_COURANT
    persona = personnalite.persona(
        config.reglage("assistant.personnalite", personnalite.DEFAUT))
    SYSTEME_COURANT = (persona + "\n\n" + SYSTEME_BASE
                       + memoire.texte_pour_systeme(memoire_courante))


# ---------------------------------------------------------------- HUD (option)

try:
    import hud
except Exception:
    hud = None

try:
    import overlay as _overlay
except Exception:
    _overlay = None

_dernier_etat_hud = None
_DERNIER_OUTIL = None
_OUTILS_MUSIQUE = {"identifier_musique", "identifier_musique_fichier", "derniere_musique"}
_RE_NOMBRE = re.compile(r"\d[\d .,:h]*\d|\d")


def _consultable(texte):
    t = (texte or "").strip()
    if len(t) > 200 or "\n" in t or len(_RE_NOMBRE.findall(t)) >= 2 or "«" in t or '"' in t:
        return True
    return False


def _afficher_overlay(texte):
    global _DERNIER_OUTIL
    outil_nom = _DERNIER_OUTIL
    _DERNIER_OUTIL = None
    if _overlay is None or not texte:
        return
    typ = "musique" if outil_nom in _OUTILS_MUSIQUE else "reponse"
    try:
        _overlay.memoriser(texte, typ)
    except Exception:
        pass
    hint = registre.affichage(outil_nom) if outil_nom else "auto"
    montrer = (hint == "toujours") or (hint != "jamais" and _consultable(texte))
    if montrer:
        try:
            _overlay.afficher(texte, type=typ)
        except Exception:
            pass


def _hud(methode, *args):
    if hud is None:
        return
    global _dernier_etat_hud
    if methode == "etat":
        if args and args[0] == _dernier_etat_hud:
            return
        _dernier_etat_hud = args[0] if args else None
    try:
        getattr(hud, methode)(*args)
    except Exception:
        pass


def _niv_hud(bloc):
    return min(1.0, niveau(bloc) / 0.2)


def _hud_status():
    try:
        from core.routage import mode_actuel
        _hud("routage", mode_actuel())
    except Exception:
        pass
    try:
        from core import budget
        e = budget.etat()
        _hud("budget", round(e["total_jour"], 2), e["plafond_jour"], round(e["pct_jour"], 3))
    except Exception:
        pass


# ---------------------------------------------------------------- audio

def niveau(bloc_float):
    return float(np.sqrt(np.mean(bloc_float**2)))


def jouer(chemin_wav):
    with wave.open(str(chemin_wav), "rb") as f:
        taux = f.getframerate()
        donnees = f.readframes(f.getnframes())
    audio = np.frombuffer(donnees, dtype=np.int16)
    sd.play(audio, samplerate=taux, device=_haut_parleur())
    sd.wait()


def bip(frequence=880, duree=0.12):
    t = np.linspace(0, duree, int(TAUX * duree), endpoint=False)
    onde = (0.25 * np.sin(2 * np.pi * frequence * t)).astype(np.float32)
    sd.play(onde, samplerate=TAUX, device=_haut_parleur())
    sd.wait()


_PROCESSUS_PAROLE = None
_INTERRUPTION = threading.Event()
_PARLE = threading.Event()
_CAPTURE_MUSIQUE = threading.Event()
_MICRO_MUET = threading.Event()


def basculer_micro(force=None):
    if force is True or (force is None and not _MICRO_MUET.is_set()):
        _MICRO_MUET.set()
    else:
        _MICRO_MUET.clear()
    muet = _MICRO_MUET.is_set()
    try:
        bip(400 if muet else 900, 0.10)
    except Exception:
        pass
    _hud("micro", muet)
    print("  [micro] " + ("coupe (mute)" if muet else "reactive"))
    return muet


def couper_parole():
    _INTERRUPTION.set()
    try:
        sd.stop()
    except Exception:
        pass
    processus = _PROCESSUS_PAROLE
    if processus is not None and processus.poll() is None:
        try:
            processus.terminate()
        except OSError:
            pass


def _jouer_audio(audio, frequence):
    if _INTERRUPTION.is_set():
        return
    sd.play(audio, samplerate=frequence, device=_haut_parleur())
    while not _INTERRUPTION.is_set():
        courant = sd.get_stream()
        if courant is None or not courant.active:
            break
        time.sleep(0.03)
    if _INTERRUPTION.is_set():
        sd.stop()


def dire(texte, interruptible=True):
    if _overlay is not None and _overlay.est_muet():
        return
    if _INTERRUPTION.is_set():
        return
    from core.tts import tts
    resultat = tts().synthetiser(texte)
    if interruptible:
        _PARLE.set()
    try:
        if resultat is not None:
            _jouer_audio(*resultat)
        else:
            _dire_sapi(texte)
    finally:
        _PARLE.clear()


def _dire_sapi(texte):
    global _PROCESSUS_PAROLE
    if _INTERRUPTION.is_set():
        return
    script = (
        "[Console]::InputEncoding = [Text.Encoding]::UTF8; "
        "$t = [Console]::In.ReadToEnd(); "
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$fr = $s.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture.Name -like 'fr*' } | "
        "Select-Object -First 1; "
        "if ($fr) { $s.SelectVoice($fr.VoiceInfo.Name) }; "
        "$s.Rate = 1; "
        "$s.Speak($t)"
    )
    processus = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _PROCESSUS_PAROLE = processus
    try:
        _, erreurs = processus.communicate(input=texte.encode("utf-8"))
        if processus.returncode and not _INTERRUPTION.is_set():
            details = (erreurs or b"").decode("utf-8", "replace").strip()
            print(f"  [SAPI] echec (code {processus.returncode}) : {details}")
    finally:
        _PROCESSUS_PAROLE = None


# ---------------------------------------------------------------- nettoyage

RESIDUS = (
    "avis", "service", "jarvis", "hey jarvis", "harvis", "arvis",
    "javis", "charvis", "chavis", "davis", "y a vis", "a vis",
    "la vis", "et vis", "ervice", "servi", "sers vis",
)

from core.hallucinations import HALLUCINATIONS  # noqa: E402

MOTS_RELANCE = (
    "attends", "attend", "arrete", "arrete-toi", "arrete toi", "stop",
    "une seconde", "deux secondes", "minute", "pardon", "non non",
)

MOTS_FIN = (
    "tais-toi", "tais toi", "chut", "silence", "ferme-la", "la ferme",
    "c'est bon", "ok merci", "d'accord merci", "laisse tomber",
)


def type_arret(texte):
    plat = sans_accents(texte.replace("’", "'").replace("‘", "'"))
    plat = "".join(c if c.isalnum() or c in " '-" else " " for c in plat)
    if any(m in plat for m in MOTS_RELANCE):
        return "relance"
    if any(m in plat for m in MOTS_FIN):
        return "fin"
    return None


def _est_oui(texte):
    return interpreter_confirmation(texte) != NON


def _est_toujours(texte):
    return interpreter_confirmation(texte) == TOUJOURS


def nettoyer(texte):
    t = texte.strip()
    plat = sans_accents(t)
    if any(h in plat for h in HALLUCINATIONS):
        return ""
    tete = None
    for sep in (",", ":", ".", "!", "?"):
        if sep in t[:20]:
            avant, _, apres = t.partition(sep)
            if len(avant.split()) <= 3:
                tete, reste = avant, apres
                break
    if tete is not None:
        if tete.strip().lower().strip("'’") in RESIDUS:
            t = reste.strip()
    mots = t.split()
    if mots and mots[0].lower().strip(",.:;!?") in RESIDUS:
        t = " ".join(mots[1:])
    return t.strip()


# ---------------------------------------------------------------- dialogue

_CONTENU_EXTERNE_TOUR = [False]


def _executer_outils(blocs):
    resultats = []
    for bloc in blocs:
        if getattr(bloc, "type", None) != "tool_use":
            continue
        nom = bloc.name
        arguments = bloc.input or {}
        outil = registre.get(nom)
        debut_outil = time.monotonic()

        if outil is None:
            resultat = f"Outil inconnu : {nom}"
        elif registre.confirmation_requise(nom, _CONTENU_EXTERNE_TOUR[0]):
            annonce = (None if outil.confirmation
                       else registre.annonce_prudence(outil, arguments))
            resultat = registre.mettre_en_attente(outil, arguments, annonce=annonce)
        else:
            try:
                resultat = registre.executer(outil, arguments)
            except Exception:
                LOG.exception("outil %s a plante", nom)
                resultat = "Desole, je n'ai pas reussi a faire ca."

        LOG.info("outil %s termine en %.3fs (type=%s)", nom,
                 time.monotonic() - debut_outil, type(resultat).__name__)
        global _DERNIER_OUTIL
        _DERNIER_OUTIL = nom

        if isinstance(resultat, dict) and resultat.get("image"):
            img = resultat["image"]
            apercu = resultat.get("apercu", "Capture d'ecran envoyee.")
            print(f"  [outil] {nom} -> {apercu}")
            _hud("outil", nom, apercu[:60])
            contenu = [{
                "type": "image",
                "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
            }]
        else:
            print(f"  [outil] {nom} -> termine")
            _hud("outil", nom, "Action terminee")
            contenu = registre.baliser_externe(nom, str(resultat))
            if nom in registre.SOURCES_EXTERNES:
                _CONTENU_EXTERNE_TOUR[0] = True

        resultats.append({
            "type": "tool_result",
            "tool_use_id": bloc.id,
            "content": contenu,
        })

        if nom in ("remember", "forget", "changer_personnalite"):
            _refaire_systeme(memoire.charger())

    return resultats


def _repondre_route_prioritaire_commune(historique):
    question = next((m.get("content") for m in reversed(historique)
                     if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
    from core.routage_intentions import decider_prioritaire
    decision = decider_prioritaire(question)
    if decision is None:
        return None

    if decision.type == "astra" and decision.tache:
        from core.routage_intentions import Decision
        decision = Decision("outil", "controle_pc_astra", {"tache": decision.tache})

    if decision.type == "astra":
        texte = "Dis-moi quelle tache tu veux que je fasse sur le PC avec Astra."
        nom_outil = "controle_pc_astra"
    elif decision.type == "vision":
        from tools.ecran import analyser_ecran
        _hud("etat", "parole")
        fil_vision = threading.Thread(target=dire, args=("Je regarde ton écran.",), daemon=True)
        fil_vision.start()
        _hud("etat", "reflexion")
        texte = analyser_ecran(decision.tache)
        fil_vision.join()
        nom_outil = "capture_screen"
    elif decision.type == "hermes":
        from tools.deleguer_a_hermes import deleguer_en_fond
        texte = deleguer_en_fond(
            decision.tache,
            intro="Hermes a terminé le travail de contenu. ",
            nom_thread="contenu-hermes",
        )
        nom_outil = "deleguer_a_hermes"
    else:
        nom_outil = decision.outil
        outil = registre.get(nom_outil)
        if outil is None:
            return None
        confirmation = outil.confirmation and not registre.est_autorise(nom_outil)
        fil_accuse = None
        if outil.lent and outil.phrase_attente and not confirmation:
            _hud("etat", "parole")
            fil_accuse = threading.Thread(target=dire, args=(outil.phrase_attente,), daemon=True)
            fil_accuse.start()
        bloc = SimpleNamespace(type="tool_use", name=nom_outil,
                               input=decision.arguments, id="route-prioritaire-commune")
        resultats = _executer_outils([bloc])
        if fil_accuse:
            fil_accuse.join()

        annonce = registre.annonce_en_attente()
        if annonce:
            _hud("etat", "parole")
            phrase = annonce + " Tu confirmes ?"
            if registre.niveau(nom_outil) == "N2":
                phrase += " Tu peux dire oui, toujours."
            if not _INTERRUPTION.is_set():
                dire(phrase, interruptible=False)
            return SENTINEL_CONFIRM
        texte = str(resultats[0]["content"] if resultats else "C'est fait.")

    historique.append({"role": "assistant", "content": texte})
    _hud("outil", nom_outil, question[:60])
    _hud("etat", "parole")
    if texte and not _INTERRUPTION.is_set():
        dire(texte)
    return texte


def repondre(historique):
    prioritaire = _repondre_route_prioritaire_commune(historique)
    if prioritaire is not None:
        return prioritaire

    fournisseur = _FOURNISSEUR_GROQ
    fil_accuse = None
    accuse_donne = False
    question = next((m.get("content") for m in reversed(historique)
                     if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
    from core.routage_intentions import modules_pour_phrase
    modules_outils = modules_pour_phrase(question)
    max_tours_outils = 6
    max_appels_outils = 12
    timeout_tour = 120.0
    debut_tour = time.monotonic()
    appels_outils = 0
    _CONTENU_EXTERNE_TOUR[0] = False

    def arreter_tour(texte):
        if fil_accuse:
            fil_accuse.join(timeout=2)
        historique.append({"role": "assistant", "content": texte})
        _hud("etat", "parole")
        if texte and not _INTERRUPTION.is_set():
            dire(texte)
        return texte

    for numero_tour in range(max_tours_outils + 1):
        if _INTERRUPTION.is_set():
            if fil_accuse:
                fil_accuse.join(timeout=2)
            return ""
        if time.monotonic() - debut_tour > timeout_tour:
            return arreter_tour("J'arrête cette demande : elle prend trop de temps.")
        try:
            debut_llm = time.monotonic()
            reponse = fournisseur.repondre(
                SYSTEME_COURANT, historique,
                registre.schemas_api(local_seulement=True, modules=modules_outils))
            LOG.info("latence LLM Groq %.3fs (tour=%s)",
                     time.monotonic() - debut_llm, numero_tour + 1)
        except Exception as e:
            print(f"  [Groq] erreur : {e}")
            return arreter_tour("Je n'arrive pas a joindre le modele pour le moment.")

        if reponse.stop_reason == "tool_use":
            _hud("etat", "reflexion")
            noms = [b.name for b in reponse.content if getattr(b, "type", None) == "tool_use"]
            if numero_tour >= max_tours_outils or appels_outils + len(noms) > max_appels_outils:
                if fil_accuse:
                    fil_accuse.join()
                return arreter_tour("J'arrête ici pour éviter une boucle d'actions.")
            appels_outils += len(noms)
            if (not accuse_donne and not _INTERRUPTION.is_set()
                    and any(n in registre.noms_lents() for n in noms)):
                accuse_donne = True
                _hud("etat", "parole")
                fil_accuse = threading.Thread(
                    target=dire, args=(registre.phrase_attente(noms),), daemon=True)
                fil_accuse.start()

            historique.append({"role": "assistant", "content": reponse.content})
            resultats = _executer_outils(reponse.content)
            historique.append({"role": "user", "content": resultats})

            annonce = registre.annonce_en_attente()
            if annonce:
                if fil_accuse:
                    fil_accuse.join()
                _hud("etat", "parole")
                phrase = annonce + " Tu confirmes ?"
                if registre.niveau(registre.nom_en_attente() or "") == "N2":
                    phrase += " Tu peux dire oui, toujours."
                if not _INTERRUPTION.is_set():
                    dire(phrase, interruptible=False)
                return SENTINEL_CONFIRM
            continue

        if fil_accuse:
            fil_accuse.join()
        texte = " ".join(
            b.text for b in reponse.content if getattr(b, "type", None) == "text"
        ).strip()
        texte = nettoyer_reponse_vocale(texte)
        historique.append({"role": "assistant", "content": texte})
        _hud("etat", "parole")
        if texte and not _INTERRUPTION.is_set():
            dire(texte)
        return texte

    return arreter_tour("Je n'ai pas réussi à terminer cette demande.")


def charger_whisper():
    print("Transcription ultra-rapide activée : Whisper Large v3 Turbo (Groq Cloud).")
    return WhisperGroqRapide()


# ---------------------------------------------------------------- principal

def _choisir_taux_capture(device):
    force = config.reglage("audio.taux", None)
    if force:
        return int(force)
    try:
        sd.check_input_settings(device=device, samplerate=TAUX, channels=1, dtype="float32")
        return TAUX
    except Exception:
        pass
    try:
        info = sd.query_devices(device, "input")
        tx = int(round(info.get("default_samplerate") or 48000))
        return tx if tx > 0 else 48000
    except Exception:
        return 48000


def _vers_16k(bloc):
    if CAPTURE_TAUX == TAUX:
        return bloc
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(TAUX, CAPTURE_TAUX)
    return resample_poly(bloc, TAUX // g, CAPTURE_TAUX // g).astype(np.float32)


def lire_bloc(flux):
    bloc, _ = flux.read(BLOC_CAPTURE)
    return _vers_16k(bloc.flatten())


def _calibrer_seuils(flux, secondes=1.0):
    niveaux = []
    t0 = time.time()
    while time.time() - t0 < secondes:
        try:
            niveaux.append(niveau(lire_bloc(flux)))
        except Exception:
            break
    if len(niveaux) < 4:
        return None
    niveaux.sort()
    fond = niveaux[len(niveaux) // 2]
    seuil_silence = min(0.05, max(0.004, fond * 2.0))
    seuil_parole = min(0.12, max(0.010, fond * 4.5))
    return seuil_silence, seuil_parole


def capturer(flux, tampon, duree_min=0.6, attente_debut=None):
    morceaux = list(tampon)
    debut = time.time()
    dernier_son = time.time()
    a_parle = False
    seuil_debut = attente_debut if attente_debut is not None else SILENCE_FIN

    while True:
        bloc = lire_bloc(flux)
        morceaux.append(bloc)
        _hud("niveau", _niv_hud(bloc))

        if niveau(bloc) > SEUIL_SILENCE:
            dernier_son = time.time()
            a_parle = True
        limite = SILENCE_FIN if a_parle else seuil_debut
        if time.time() - dernier_son > limite:
            break
        if time.time() - debut > DUREE_MAX:
            print("  (trop long, je coupe)")
            break

    tampon.clear()
    audio = np.concatenate(morceaux)
    return audio if len(audio) >= TAUX * duree_min else None


def attendre_suite(flux, tampon, duree=DUREE_SUITE):
    _hud("etat", "ecoute")
    tampon.clear()
    debut = time.time()
    blocs_voix = 0
    while time.time() - debut < duree:
        try:
            bloc = lire_bloc(flux)
        except Exception:
            return False
        tampon.append(bloc)
        _hud("niveau", _niv_hud(bloc))
        if niveau(bloc) > SEUIL_PAROLE_SUR:
            blocs_voix += 1
            if blocs_voix >= 3:
                return True
        else:
            blocs_voix = 0
    return False


def repondre_en_ecoutant(historique, flux, reveil, whisper):
    _INTERRUPTION.clear()
    resultat = {}

    def travail():
        try:
            resultat["texte"] = repondre(historique)
        except Exception as e:
            resultat["erreur"] = e

    thread = threading.Thread(target=travail, daemon=True)
    thread.start()

    interrompu = False
    relancer = False

    facteur = float(config.reglage("interruption.facteur", 1.8))
    seuil_min = float(config.reglage("interruption.seuil", SEUIL_PAROLE_SUR))
    blocs_requis = int(config.reglage("interruption.blocs", BLOCS_AVANT_VERIF))

    base = None
    tampon = []
    blocs_sur = 0
    derniere_verif = 0.0

    while thread.is_alive():
        if _CAPTURE_MUSIQUE.is_set():
            time.sleep(0.05)
            continue
        try:
            bloc = lire_bloc(flux)
        except Exception:
            if _CAPTURE_MUSIQUE.is_set():
                time.sleep(0.05)
                continue
            break
        _hud("niveau", _niv_hud(bloc))

        if not _PARLE.is_set():
            base = None
            blocs_sur = 0
            tampon = []
            continue

        scores = reveil.predict((bloc * 32767).astype(np.int16))
        if max(scores.values()) >= SEUIL_INTERRUPTION:
            couper_parole()
            interrompu, relancer = True, True
            print("  [micro] Je me tais.")
            break

        niv = niveau(bloc)
        if base is None:
            base = niv
        seuil_sur = max(seuil_min, base * facteur)
        base = 0.97 * base + 0.03 * niv

        if niv > seuil_sur:
            tampon.append(bloc)
            blocs_sur += 1
        else:
            if 0 < blocs_sur < blocs_requis:
                tampon = []
            blocs_sur = 0

        maintenant = time.time()
        if blocs_sur >= blocs_requis and maintenant - derniere_verif > DELAI_ENTRE_VERIFS:
            derniere_verif = maintenant
            extrait = np.concatenate(tampon[-30:])
            tampon = []
            blocs_sur = 0
            try:
                segments, _ = whisper.transcribe(extrait, language="fr", beam_size=1)
                dit = " ".join(s.text for s in segments).strip()
            except Exception:
                dit = ""
            categorie = type_arret(dit) if dit else None
            if categorie:
                couper_parole()
                interrompu = True
                relancer = (categorie == "relance")
                print(f"  [micro] Interruption : {dit}")
                break

    thread.join(timeout=10)
    reveil.reset()

    if "erreur" in resultat:
        raise resultat["erreur"]

    return resultat.get("texte", ""), interrompu, relancer


def _confirmer(interrompu, relancer, whisper, historique, flux):
    if interrompu:
        registre.annuler_confirme()
        return "", relancer

    _INTERRUPTION.clear()
    _hud("etat", "ecoute")
    audio_conf = capturer(flux, deque(), duree_min=0.3,
                          attente_debut=float(config.reglage("assistant.attente_confirmation", 3.0)))
    reponse = ""
    if audio_conf is not None:
        seg, _ = whisper.transcribe(audio_conf, language="fr", beam_size=5)
        reponse = nettoyer(" ".join(s.text for s in seg).strip())
    print(f"  [confirmation] {reponse or '(rien)'}")

    memoriser = _est_toujours(reponse)
    if _est_oui(reponse) or memoriser:
        res = registre.executer_confirme(memoriser=memoriser)
    else:
        registre.annuler_confirme()
        res = "D'accord, j'annule."

    _hud("etat", "parole")
    if res and not _INTERRUPTION.is_set():
        dire(res)
    historique.append({"role": "assistant", "content": res})
    return res, False


def _tronquer(historique):
    if len(historique) > 40:
        del historique[:len(historique) - 40]
        while historique and not (
            historique[0]["role"] == "user"
            and isinstance(historique[0]["content"], str)
        ):
            historique.pop(0)


def traiter(audio, whisper, historique, flux, reveil):
    debut_stt = time.monotonic()
    segments, _ = whisper.transcribe(audio, language="fr", beam_size=5)
    question = nettoyer(" ".join(s.text for s in segments).strip())
    LOG.info("latence STT %.3fs", time.monotonic() - debut_stt)

    if not question or len(question) < 3:
        print("  (rien compris)\n")
        return False

    print(f"  Vous : {question}")
    _hud("dire_vous", question)
    _hud("etat", "reflexion")
    historique.append({"role": "user", "content": question})

    texte, interrompu, relancer = repondre_en_ecoutant(historique, flux, reveil, whisper)

    if texte == SENTINEL_CONFIRM:
        _hud("confirmation", True)
        texte, relancer = _confirmer(interrompu, relancer, whisper, historique, flux)
        _hud("confirmation", False)
    elif not texte:
        texte = "C'est fait."
        if not interrompu:
            _hud("etat", "parole")
            dire(texte)

    _hud("dire_jarvis", texte)
    _afficher_overlay(texte)
    _hud_status()
    print(f"  Jarvis : {texte}\n")
    _tronquer(historique)
    return relancer


def _feedback_geste(geste):
    freq = 1200 if geste == "armement" or geste.startswith("mode_") else 900
    try:
        threading.Thread(target=lambda: bip(freq, 0.05), daemon=True).start()
    except Exception:
        pass
    _hud("outil", "geste", geste)


def _installer_raccourci_gestes():
    combo = config.reglage("gestes.raccourci", "ctrl+alt+g")
    combo_demo = config.reglage("gestes.raccourci_demo", "ctrl+alt+d")
    combo_regard = config.reglage("gestes.raccourci_regard", "ctrl+alt+r")
    if not combo and not combo_demo and not combo_regard:
        return
    try:
        import keyboard
    except Exception:
        return
    from core import gestes

    def _toggle():
        print(gestes.arreter() if gestes.actif() else gestes.demarrer())

    def _demo():
        print(gestes.demarrer_demo())

    def _regard():
        print(gestes.arreter_regard() if gestes.regard_actif() else gestes.demarrer_regard())

    try:
        if combo:
            keyboard.add_hotkey(combo, _toggle)
        if combo_demo:
            keyboard.add_hotkey(combo_demo, _demo)
        if combo_regard:
            keyboard.add_hotkey(combo_regard, _regard)
    except Exception:
        LOG.exception("gestes: raccourci clavier")


def _installer_raccourci_micro():
    combo = config.reglage("audio.raccourci_mute", "ctrl+alt+m")
    if not combo:
        return
    try:
        import keyboard
        keyboard.add_hotkey(combo, basculer_micro)
    except Exception:
        pass


def main():
    print("Chargement des modeles (Version Groq Turbo)...")

    serveur_sans_peripheriques = bool(
        config.reglage("poste_principal.actif", False)
        and config.reglage("poste_principal.serveur_sans_peripheriques", False))

    registre.charger_outils()
    voix.definir_parleur(dire)
    try:
        from tools.alexa import precharger_routines
        precharger_routines()
    except Exception:
        pass

    reveil = None
    if not serveur_sans_peripheriques:
        reveil = WakeModel(wakeword_model_paths=[str(
            Path(openwakeword.__file__).parent / "resources" / "models" / "hey_jarvis_v0.1.onnx"
        )])

    whisper = charger_whisper()

    from tools.appels import definir_transcripteur
    definir_transcripteur(lambda chemin: " ".join(
        s.text for s in whisper.transcribe(chemin, language="fr", beam_size=5)[0]).strip())
    from tools.appel_direct import definir_transcripteur_direct
    definir_transcripteur_direct(lambda audio: " ".join(
        s.text for s in whisper.transcribe(audio, language="fr", beam_size=1)[0]).strip())

    charger_pieces_hue()
    allumer_si_nuit()

    from tools.presence import demarrer_presence
    demarrer_presence()

    from tools.discord_bot import demarrer_discord
    demarrer_discord()

    from tools.instagram import demarrer_refresh_instagram
    demarrer_refresh_instagram()

    if (config.reglage("serveur.actif", False) or config.reglage("pont_iphone.actif", False)
            or config.reglage("gestes.actif", False) or config.reglage("cockpit.actif", False)
            or (config.reglage("satellites", []) or [])
            or (config.reglage("desktop_agents", []) or [])):
        from core.serveur import demarrer as demarrer_serveur_web
        demarrer_serveur_web()
        if config.reglage("cockpit.actif", False) and not serveur_sans_peripheriques:
            try:
                from core import cockpit
                threading.Thread(target=cockpit.ouvrir_fenetre, daemon=True).start()
            except Exception:
                pass

    try:
        import atexit
        from core import gestes
        gestes.definir_hooks(couper_tts=couper_parole, feedback=_feedback_geste)
        atexit.register(gestes.arreter_regard)
        atexit.register(gestes.arreter)
        if config.reglage("gestes.actif", False) and not serveur_sans_peripheriques:
            print(gestes.demarrer())
        if not serveur_sans_peripheriques:
            _installer_raccourci_gestes()
            _installer_raccourci_micro()
    except Exception:
        pass

    _fournisseur = _FOURNISSEUR_GROQ
    print(f"Mode : Turbo Groq — LLM {_fournisseur.modele}, "
          f"TTS {__import__('core.tts', fromlist=['tts']).tts().nom}.")

    if not serveur_sans_peripheriques:
        _hud("demarrer", True, str(config.reglage("hud.fenetre", "app") or "app"),
             config.reglage("hud.ecran", None))
        _hud("config", f"Groq · {_fournisseur.modele}", "whisper-v3-turbo")
        _hud_status()
        try:
            from tools import deleguer_a_hermes as _dh
            threading.Thread(target=_dh.rafraichir_hud, daemon=True).start()
        except Exception:
            pass

    faits = memoire.charger()
    if faits:
        print(f"Memoire : {len(faits)} information(s).")
    _refaire_systeme(faits)
    historique = []

    global CAPTURE_TAUX, BLOC_CAPTURE
    CAPTURE_TAUX = _choisir_taux_capture(MICRO)
    BLOC_CAPTURE = int(round(BLOC * CAPTURE_TAUX / TAUX))
    flux = sd.InputStream(
        samplerate=CAPTURE_TAUX, channels=1, dtype="float32",
        device=MICRO, blocksize=BLOC_CAPTURE,
    )
    flux.start()

    if (config.reglage("assistant.auto_calibration", True)
            and config.reglage("assistant.seuil_silence", None) is None
            and config.reglage("assistant.seuil_parole", None) is None):
        global SEUIL_SILENCE, SEUIL_PAROLE_SUR
        cal = _calibrer_seuils(flux)
        if cal:
            SEUIL_SILENCE, SEUIL_PAROLE_SUR = cal
            print(f"[audio] seuils auto-calibrés : silence={SEUIL_SILENCE:.4f} parole={SEUIL_PAROLE_SUR:.4f}")

    if _overlay is not None and config.reglage("overlay.actif", True):
        try:
            _overlay.demarrer({
                "actif": True,
                "muet": config.reglage("overlay.muet_visuel", False),
                "ecran": config.reglage("overlay.ecran", 1),
                "coin": config.reglage("overlay.coin", "bas-droite"),
                "opacite": config.reglage("overlay.opacite", 0.92),
                "largeur": config.reglage("overlay.largeur", 420),
                "duree_min": config.reglage("overlay.duree_min", 4.0),
                "duree_max": config.reglage("overlay.duree_max", 14.0),
                "marge": config.reglage("overlay.marge", 24),
                "exclure_obs": config.reglage("overlay.exclure_obs", True),
            })
        except Exception:
            pass

    print('\nPret. Dites "Hey Jarvis". Ctrl+C pour quitter.\n')

    tampon = deque(maxlen=6)
    enchainer = False

    try:
        while True:
            suite = enchainer
            if not enchainer:
                bloc = lire_bloc(flux)
                tampon.append(bloc)

                if _MICRO_MUET.is_set():
                    _hud("etat", "muet")
                    continue

                _hud("etat", "veille")
                _hud("niveau", _niv_hud(bloc))

                scores = reveil.predict((bloc * 32767).astype(np.int16))
                score_reveil = max(scores.values())
                if score_reveil < SEUIL_REVEIL:
                    continue
                reveil.reset()
                from core.arbitrage_micro import reserver_reveil
                priorite = float(config.reglage("assistant.priorite_micro", 1.0) or 1.0)
                if not reserver_reveil("principal", score_reveil * priorite):
                    continue

            enchainer = False
            _hud("etat", "ecoute")
            if not suite:
                print("  [micro] Oui ?")
                if config.reglage("hud.premier_plan_au_reveil", True):
                    threading.Thread(target=_hud, args=("mettre_au_premier_plan",), daemon=True).start()
                bip()

            audio = capturer(flux, tampon)
            if audio is None:
                print("  (rien entendu)\n")
                continue

            if traiter(audio, whisper, historique, flux, reveil):
                enchainer = True
                continue

            print(f"  [micro] J'ecoute encore {int(DUREE_SUITE)} s...")
            enchainer = attendre_suite(flux, tampon)
            if not enchainer:
                _hud("etat", "veille")

    except KeyboardInterrupt:
        print("\nAu revoir.")
    finally:
        couper_parole()
        flux.stop()
        flux.close()


if __name__ == "__main__":
    main()
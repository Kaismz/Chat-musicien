"""
Tokenizer musical REMI-like pour le projet Chat Musicien.

Représentation :
  - BAR : marque le début d'une mesure (4 beats = 4/4)
  - POSITION_<i> : position dans la mesure, i in [0, STEPS_PER_BAR)
  - NOTE_ON_<p> : pitch MIDI, p in [PITCH_MIN, PITCH_MAX]
  - DURATION_<d> : durée de la note en pas de grille, d in [1, MAX_DURATION_STEPS]
  - VELOCITY_<v> : vélocité quantifiée, v in [0, N_VELOCITY_BINS-1]

Tokens spéciaux : <PAD>, <BOS>, <EOS>, <UNK>
"""

from pathlib import Path
from miditoolkit import MidiFile, Instrument, Note
import json
import pickle


# ---------------------------------------------------------------------------
# Configuration de la grille musicale
# ---------------------------------------------------------------------------
# Décisions issues de l'exploration du dataset :
PITCH_MIN = 21          # A0 (note la plus basse du piano)
PITCH_MAX = 108         # C8 (note la plus haute du piano)

STEPS_PER_BEAT = 4      # 1/16 de beat = double-croche
BEATS_PER_BAR = 4       # mesure 4/4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR   # 16 positions par mesure

MAX_DURATION_STEPS = 16 # durée max = 4 beats = 1 mesure
N_VELOCITY_BINS = 32    # quantification de la vélocité (0..127 -> 0..31)

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


class MusicTokenizer:
    """Tokenizer REMI-like : MIDI <-> séquence d'IDs entiers."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()

    # ------------------------------------------------------------------
    # Construction du vocabulaire (déterministe, pas besoin de fit)
    # ------------------------------------------------------------------
    def _build_vocab(self):
        tokens = list(SPECIAL_TOKENS)
        tokens.append("BAR")
        tokens += [f"POSITION_{i}" for i in range(STEPS_PER_BAR)]
        tokens += [f"NOTE_ON_{p}" for p in range(PITCH_MIN, PITCH_MAX + 1)]
        tokens += [f"DURATION_{d}" for d in range(1, MAX_DURATION_STEPS + 1)]
        tokens += [f"VELOCITY_{v}" for v in range(N_VELOCITY_BINS)]

        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self):
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self):
        return self.token_to_id["<EOS>"]

    # ------------------------------------------------------------------
    # MIDI -> tokens (encode)
    # ------------------------------------------------------------------
    def encode_midi(self, midi_path):
        """
        Lit un fichier MIDI et renvoie une liste d'IDs de tokens.
        Retourne None si le fichier est inexploitable.
        """
        try:
            mid = MidiFile(str(midi_path))
        except Exception:
            return None

        tpb = mid.ticks_per_beat
        ticks_per_step = tpb / STEPS_PER_BEAT       # 1 step = 1/16 beat

        # On regroupe les notes de tous les instruments (piano principalement)
        all_notes = []
        for inst in mid.instruments:
            if inst.is_drum:    # on ignore les pistes de batterie
                continue
            all_notes.extend(inst.notes)

        if not all_notes:
            return None

        # Tri par (start, pitch) pour avoir un ordre canonique
        all_notes.sort(key=lambda n: (n.start, n.pitch))

        # Quantification : on calcule pour chaque note sa position en "steps"
        events = []
        for n in all_notes:
            step = round(n.start / ticks_per_step)
            duration_steps = max(1, round((n.end - n.start) / ticks_per_step))
            duration_steps = min(duration_steps, MAX_DURATION_STEPS)
            pitch = max(PITCH_MIN, min(PITCH_MAX, n.pitch))
            vel_bin = min(N_VELOCITY_BINS - 1, n.velocity * N_VELOCITY_BINS // 128)
            events.append((step, pitch, duration_steps, vel_bin))

        # Génération des tokens : on insère BAR / POSITION quand le bar/pos change
        tokens = ["<BOS>"]
        last_bar = -1
        last_pos = -1
        for step, pitch, dur, vel in events:
            bar_idx = step // STEPS_PER_BAR
            pos_in_bar = step % STEPS_PER_BAR

            if bar_idx != last_bar:
                # Insère un BAR (et autant de BAR vides que nécessaire si silence long)
                # Simplification : un seul BAR par changement (le modèle apprendra)
                tokens.append("BAR")
                last_bar = bar_idx
                last_pos = -1

            if pos_in_bar != last_pos:
                tokens.append(f"POSITION_{pos_in_bar}")
                last_pos = pos_in_bar

            tokens.append(f"NOTE_ON_{pitch}")
            tokens.append(f"DURATION_{dur}")
            tokens.append(f"VELOCITY_{vel}")

        tokens.append("<EOS>")

        # Conversion en IDs
        ids = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]
        return ids

    # ------------------------------------------------------------------
    # tokens -> MIDI (decode)
    # ------------------------------------------------------------------
    def decode_to_midi(self, token_ids, output_path, tempo=120, ticks_per_beat=384):
        """
        Reconvertit une séquence de tokens en fichier MIDI.
        """
        ticks_per_step = ticks_per_beat // STEPS_PER_BEAT

        notes = []
        current_bar = 0
        current_pos = 0
        pending_pitch = None
        pending_duration = None

        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), "<UNK>")

            if tok == "BAR":
                current_bar += 1
                current_pos = 0
            elif tok.startswith("POSITION_"):
                current_pos = int(tok.split("_")[1])
                pending_pitch = None
                pending_duration = None
            elif tok.startswith("NOTE_ON_"):
                pending_pitch = int(tok.split("_")[2])
                pending_duration = None
            elif tok.startswith("DURATION_"):
                pending_duration = int(tok.split("_")[1])
            elif tok.startswith("VELOCITY_"):
                if pending_pitch is not None and pending_duration is not None:
                    vel_bin = int(tok.split("_")[1])
                    velocity = min(127, (vel_bin * 128) // N_VELOCITY_BINS + 4)
                    abs_step = current_bar * STEPS_PER_BAR + current_pos
                    start = abs_step * ticks_per_step
                    end = start + pending_duration * ticks_per_step
                    notes.append(Note(velocity=velocity, pitch=pending_pitch,
                                      start=start, end=end))
                    pending_pitch = None
                    pending_duration = None
            # On ignore <PAD>, <BOS>, <EOS>, <UNK>

        # Construction du MIDI
        mid = MidiFile()
        mid.ticks_per_beat = ticks_per_beat
        from miditoolkit.midi.containers import TempoChange
        mid.tempo_changes = [TempoChange(tempo=tempo, time=0)]
        piano = Instrument(program=0, is_drum=False, name="Piano")
        piano.notes = notes
        mid.instruments = [piano]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mid.dump(str(output_path))
        return len(notes)

    # ------------------------------------------------------------------
    # Sauvegarde / chargement (pour Colab)
    # ------------------------------------------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load(cls, path):
        tok = cls()  # vocab déterministe, on vérifie juste qu'il match
        with open(path) as f:
            saved = json.load(f)
        assert saved == tok.token_to_id, "Vocab incompatible"
        return tok
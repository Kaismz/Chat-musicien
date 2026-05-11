"""
Prétraitement : tokenise tous les MIDI du dataset et sauvegarde en .pkl

Usage :
    python -m src.preprocess

Sortie :
    outputs/tokens/train_tokens.pkl
    outputs/tokens/val_tokens.pkl
    outputs/tokens/vocab.json
"""

from pathlib import Path
import pickle
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.tokenizer import MusicTokenizer


# ---------------------------------------------------------------------------
# Worker pour le multi-process
# ---------------------------------------------------------------------------
# Note : on instancie le tokenizer dans chaque process (vocab déterministe,
# donc identique partout).
def _tokenize_one(path_str):
    tok = MusicTokenizer()
    ids = tok.encode_midi(path_str)
    if ids is None:
        return None
    return ids


# ---------------------------------------------------------------------------
# Filtres de qualité
# ---------------------------------------------------------------------------
MIN_TOKENS = 64        # rejet des morceaux quasi vides
MAX_TOKENS_HARD = 50_000  # rejet UNIQUEMENT des aberrations vraiment énormes
TRUNCATE_AT = 16_384   # on tronque les longs morceaux ici 


def main(data_dir, output_dir, val_ratio=0.05, max_files=None, n_workers=None, seed=42):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lister les MIDI
    midi_files = sorted(list(data_dir.rglob("*.mid")) + list(data_dir.rglob("*.midi")))
    print(f"MIDI trouvés : {len(midi_files)}")
    if max_files:
        random.Random(seed).shuffle(midi_files)
        midi_files = midi_files[:max_files]
        print(f"Limite : {max_files} fichiers traités")

    # 2. Tokenisation parallèle
    all_sequences = []
    failed = 0

    n_workers = n_workers or max(1, (Path().stat().st_nlink or 1))  # heuristique
    print(f"Tokenisation avec {n_workers} processus...")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_tokenize_one, str(p)): p for p in midi_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tokenisation"):
            ids = fut.result()
            if ids is None:
                failed += 1
                continue
            if len(ids) < MIN_TOKENS:
                failed += 1
                continue
            if len(ids) > MAX_TOKENS_HARD:
                failed += 1   # vraie aberration (fichier corrompu)
                continue
            if len(ids) > TRUNCATE_AT:
                ids = ids[:TRUNCATE_AT]   # on tronque, on garde
            all_sequences.append(ids)

    print(f"\nSéquences valides   : {len(all_sequences):,}")
    print(f"Séquences rejetées  : {failed:,}")
    if all_sequences:
        lengths = [len(s) for s in all_sequences]
        print(f"Longueurs : min={min(lengths)}, médiane={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")
        print(f"Total tokens : {sum(lengths):,}")

    # 3. Split train/val
    rng = random.Random(seed)
    rng.shuffle(all_sequences)
    n_val = max(1, int(len(all_sequences) * val_ratio))
    val_seqs = all_sequences[:n_val]
    train_seqs = all_sequences[n_val:]
    print(f"\nTrain : {len(train_seqs):,}  |  Val : {len(val_seqs):,}")

    # 4. Sauvegardes
    with open(output_dir / "train_tokens.pkl", "wb") as f:
        pickle.dump(train_seqs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "val_tokens.pkl", "wb") as f:
        pickle.dump(val_seqs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Vocab (sécurité, même si déterministe)
    tok = MusicTokenizer()
    tok.save(output_dir / "vocab.json")

    print(f"\n✅ Sauvegardé dans {output_dir.resolve()}")
    print(f"   - train_tokens.pkl  ({(output_dir/'train_tokens.pkl').stat().st_size / 1e6:.1f} Mo)")
    print(f"   - val_tokens.pkl    ({(output_dir/'val_tokens.pkl').stat().st_size / 1e6:.1f} Mo)")
    print(f"   - vocab.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/GrandMidiPiano")
    parser.add_argument("--output_dir", default="outputs/tokens")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_files", type=int, default=None,
                        help="Limite le nb de fichiers (test rapide)")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Nb de processus parallèles")
    args = parser.parse_args()

    main(args.data_dir, args.output_dir, args.val_ratio,
         args.max_files, args.n_workers)
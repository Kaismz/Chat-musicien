"""
Génère les 5 MIDI demandés pour le rendu, avec différentes stratégies.

Usage :
    python -m src.generate_batch --model outputs/models/chat_musicien_best.pth
"""

import argparse
from pathlib import Path
import random

import torch

from src.gpt import GPTModel, GPT_CONFIG_MUSIC
from src.tokenizer import MusicTokenizer
from src.generate import generate, make_prompt


# 5 stratégies de génération pour montrer la variété
STRATEGIES = [
    {
        "name": "01_creation_libre",
        "desc": "Génération depuis <BOS>, T=1.0, top_k=30",
        "prompt_midi": None,
        "max_tokens": 800,
        "temperature": 1.0,
        "top_k": 30,
        "top_p": 0.0,
        "tempo": 110,
    },
    {
        "name": "02_temperature_basse",
        "desc": "Conservateur, T=0.7, top_k=15 (le modèle prend peu de risques)",
        "prompt_midi": None,
        "max_tokens": 800,
        "temperature": 0.7,
        "top_k": 15,
        "top_p": 0.0,
        "tempo": 100,
    },
    {
        "name": "03_temperature_haute",
        "desc": "Créatif, T=1.2, top_p=0.95 (nucleus, varié)",
        "prompt_midi": None,
        "max_tokens": 800,
        "temperature": 1.2,
        "top_k": 0,
        "top_p": 0.95,
        "tempo": 120,
    },
    {
        "name": "04_continuation_morceau_1",
        "desc": "Le modèle continue un morceau réel (prompt MIDI #1)",
        "prompt_midi": "auto_1",        # sera choisi auto dans le dataset
        "max_tokens": 700,
        "temperature": 1.0,
        "top_k": 30,
        "top_p": 0.0,
        "tempo": 110,
    },
    {
        "name": "05_continuation_morceau_2",
        "desc": "Le modèle continue un autre morceau réel (prompt MIDI #2)",
        "prompt_midi": "auto_2",
        "max_tokens": 700,
        "temperature": 0.9,
        "top_k": 25,
        "top_p": 0.0,
        "tempo": 105,
    },
]


def pick_random_midi(data_dir, seed):
    """Choisit un MIDI au hasard pour servir de prompt."""
    midi_files = list(Path(data_dir).rglob("*.mid"))
    rng = random.Random(seed)
    return str(rng.choice(midi_files))


def main(args):
   # Reproductibilité
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device)
    tokenizer = MusicTokenizer()

    # Charge le modèle une fois
    model = GPTModel(GPT_CONFIG_MUSIC).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Modèle chargé : {args.model}")
    print(f"Génération sur {device}")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, strat in enumerate(STRATEGIES, 1):
        print(f"\n[{i}/5] {strat['name']}")
        print(f"     {strat['desc']}")

        # Choix du prompt
        prompt_midi = strat["prompt_midi"]
        if prompt_midi == "auto_1":
            prompt_midi = pick_random_midi(args.data_dir, seed=i * 13)
        elif prompt_midi == "auto_2":
            prompt_midi = pick_random_midi(args.data_dir, seed=i * 17)
        if prompt_midi:
            print(f"     Prompt MIDI : {Path(prompt_midi).name}")

        prompt = make_prompt(tokenizer, prompt_midi)

        out = generate(
            model, prompt,
            max_new_tokens=strat["max_tokens"],
            context_length=GPT_CONFIG_MUSIC["context_length"],
            temperature=strat["temperature"],
            top_k=strat["top_k"] if strat["top_k"] > 0 else None,
            top_p=strat["top_p"] if strat["top_p"] > 0 else None,
            eos_id=None,    # on génère pile max_tokens
            device=device,
        )

        out_ids = out[0].cpu().tolist()
        output_path = output_dir / f"{strat['name']}.mid"
        n_notes = tokenizer.decode_to_midi(out_ids, str(output_path),
                                           tempo=strat["tempo"])
        print(f"     ✅ {output_path.name} : {n_notes} notes")

    print("\n" + "=" * 60)
    print(f"✅ 5 fichiers MIDI générés dans {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model",     required=True)
    parser.add_argument("--output_dir", default="outputs/generated")
    parser.add_argument("--data_dir",   default="data/GrandMidiPiano",
                        help="Pour les prompts MIDI auto")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()
    main(args)
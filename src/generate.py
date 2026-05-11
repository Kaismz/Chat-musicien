"""
Génération musicale avec le Chat Musicien entraîné.

Usage :
    python -m src.generate \
        --model outputs/models/chat_musicien_best.pth \
        --output outputs/generated/song1.mid \
        --max_tokens 800 --temperature 1.0 --top_k 30

Modes :
    1. Génération libre : commence par <BOS>, modèle invente tout
    2. Génération conditionnée : donne un MIDI prompt → continuation
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.gpt import GPTModel, GPT_CONFIG_MUSIC
from src.tokenizer import MusicTokenizer


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens, context_length,
             temperature=1.0, top_k=None, top_p=None,
             eos_id=None, device="cpu"):
    """
    Génération autoregressive avec temperature + top-k/top-p sampling.

    Args:
        model         : GPTModel chargé en eval mode
        idx           : tensor (1, seq_len) avec les tokens du prompt
        max_new_tokens: nb de tokens à générer
        context_length: fenêtre du modèle (256 dans notre cas)
        temperature   : T>1 = créatif, T<1 = conservateur, T=0 = argmax
        top_k         : si défini, ne considère que les k tokens les plus probables
        top_p         : nucleus sampling (alternative à top_k)
        eos_id        : si rencontré, on stoppe (None pour générer pile max_new_tokens)
    """
    model.eval()
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        # On tronque le contexte à context_length
        idx_cond = idx[:, -context_length:]
        logits = model(idx_cond)               # (1, T, V)
        logits = logits[:, -1, :]              # (1, V) — dernier token seulement

        # Temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        elif temperature == 0:
            # Mode déterministe (argmax)
            next_id = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            continue

        # Top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k)
            min_v = v[:, [-1]]
            logits = torch.where(logits < min_v,
                                 torch.full_like(logits, -float("inf")),
                                 logits)

        # Top-p (nucleus)
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = sorted_probs.cumsum(dim=-1)
            # Tokens dont la proba cumulée dépasse top_p sont écartés
            mask = cum_probs > top_p
            mask[:, 0] = False  # garde toujours au moins 1 token
            sorted_logits[mask] = -float("inf")
            # Re-trie en place originale
            logits = torch.full_like(logits, -float("inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)

        # Échantillonnage
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    return idx


# ---------------------------------------------------------------------------
# Préparation du prompt
# ---------------------------------------------------------------------------
def make_prompt(tokenizer, prompt_midi=None):
    """
    Construit le prompt initial.
    - Si prompt_midi fourni : encode les premières mesures du MIDI
    - Sinon : commence par <BOS>
    """
    if prompt_midi is None:
        return torch.tensor([[tokenizer.bos_id]], dtype=torch.long)

    ids = tokenizer.encode_midi(prompt_midi)
    if ids is None:
        print(f"⚠️  Échec encodage de {prompt_midi}, fallback sur <BOS>")
        return torch.tensor([[tokenizer.bos_id]], dtype=torch.long)

    # On garde les ~30 premiers tokens (~5-6 notes) pour amorcer
    PROMPT_LEN = 30
    ids = ids[:PROMPT_LEN]
    return torch.tensor([ids], dtype=torch.long)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device(args.device)
    tokenizer = MusicTokenizer()

    # Modèle
    model = GPTModel(GPT_CONFIG_MUSIC).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Modèle chargé : {args.model}")
    print(f"Paramètres : {model.num_params():,}")

    # Prompt
    prompt = make_prompt(tokenizer, args.prompt_midi)
    print(f"Prompt : {prompt.shape[1]} tokens")
    print(f"  → {[tokenizer.id_to_token[int(i)] for i in prompt[0][:10]]}...")

    # Génération
    print(f"\nGénération de {args.max_tokens} tokens "
          f"(T={args.temperature}, top_k={args.top_k}, top_p={args.top_p})...")
    out = generate(
        model, prompt,
        max_new_tokens=args.max_tokens,
        context_length=GPT_CONFIG_MUSIC["context_length"],
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p > 0 else None,
        eos_id=tokenizer.eos_id if args.stop_on_eos else None,
        device=device,
    )

    out_ids = out[0].cpu().tolist()
    print(f"Tokens générés : {len(out_ids)}")
    preview = [tokenizer.id_to_token.get(i, "?") for i in out_ids[:20]]
    print(f"Aperçu : {preview}")

    # Sauvegarde MIDI
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_notes = tokenizer.decode_to_midi(out_ids, str(output_path),
                                       tempo=args.tempo)
    print(f"\n✅ MIDI sauvegardé : {output_path.resolve()}")
    print(f"   {n_notes} notes sur ~{n_notes / 4:.0f} secondes (à {args.tempo} BPM)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,
                        help="Chemin du .pth")
    parser.add_argument("--output",      default="outputs/generated/song.mid")
    parser.add_argument("--prompt_midi", default=None,
                        help="MIDI à utiliser comme prompt (optionnel)")
    parser.add_argument("--max_tokens",  type=int, default=800)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k",       type=int, default=30)
    parser.add_argument("--top_p",       type=float, default=0.0)
    parser.add_argument("--tempo",       type=int, default=110)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--stop_on_eos", action="store_true",
                        help="Arrêter dès que <EOS> est généré")
    args = parser.parse_args()
    main(args)
"""
Boucle d'entraînement du Chat Musicien.

Usage local (test rapide CPU) :
    python -m src.train --steps 100

Usage Colab (full training) :
    python -m src.train --epochs 10 --batch_size 64 --device cuda

Sauvegarde :
    outputs/models/chat_musicien_best.pth   (meilleur val loss)
    outputs/models/chat_musicien_last.pth   (dernier checkpoint)
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.gpt import GPTModel, GPT_CONFIG_MUSIC
from src.dataset import make_dataloaders


# ---------------------------------------------------------------------------
# Évaluation : loss moyenne sur le val_loader
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.flatten(0, 1), y.flatten())
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def train(args):
    # 1. Device
    device = torch.device(args.device)
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    # 2. DataLoaders
    train_loader, val_loader, train_ds, val_ds = make_dataloaders(
        args.train_pkl, args.val_pkl,
        context_length=GPT_CONFIG_MUSIC["context_length"],
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train : {len(train_ds):,} fenêtres  |  Val : {len(val_ds):,} fenêtres")
    print(f"Batches / epoch : {len(train_loader):,}")

    # 3. Modèle
    model = GPTModel(GPT_CONFIG_MUSIC).to(device)
    print(f"Paramètres : {model.num_params():,} (~{model.num_params()/1e6:.2f} M)")

    if args.resume:
        print(f"Reprise depuis {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 4. Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01, betas=(0.9, 0.95))

    # Warmup linéaire puis cosine decay (recette GPT classique)
    total_steps = args.epochs * len(train_loader) if args.steps == 0 else args.steps
    warmup_steps = min(500, total_steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # 5. Boucle
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    global_step = 0
    t0 = time.time()
    log = {"steps": [], "train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits.flatten(0, 1), y.flatten())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Log régulier
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1}/{args.epochs} | step {global_step} | "
                      f"loss {loss.item():.4f} | lr {lr_now:.2e} | "
                      f"{elapsed:.1f}s")

            # Évaluation + checkpoint
            if global_step % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                ppl = math.exp(min(val_loss, 20))   # perplexity
                print(f"  >> val_loss {val_loss:.4f} | perplexity {ppl:.2f}")
                log["steps"].append(global_step)
                log["train_loss"].append(loss.item())
                log["val_loss"].append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), output_dir / "chat_musicien_best.pth")
                    print(f"  ✅ nouveau best, sauvegardé")

            # Limite manuelle pour test rapide
            if args.steps and global_step >= args.steps:
                break
        if args.steps and global_step >= args.steps:
            break

    # 6. Sauvegardes finales
    torch.save(model.state_dict(), output_dir / "chat_musicien_last.pth")
    import json
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ Entraînement terminé en {(time.time()-t0)/60:.1f} min")
    print(f"   best val loss : {best_val:.4f}  |  perplexity : {math.exp(min(best_val,20)):.2f}")
    print(f"   modèle sauvegardé dans {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pkl", default="outputs/tokens/train_tokens.pkl")
    parser.add_argument("--val_pkl",   default="outputs/tokens/val_tokens.pkl")
    parser.add_argument("--output_dir", default="outputs/models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps",  type=int, default=0,
                        help="Si > 0, limite le nb de steps total (test rapide)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every",  type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--resume", default="",
                        help="Chemin d'un .pth pour reprendre l'entraînement")
    args = parser.parse_args()
    train(args)
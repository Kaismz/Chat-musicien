"""
Dataset PyTorch pour l'entraînement du Chat Musicien.

Charge les .pkl de tokens prétokenisés et fournit des paires (input, target)
de longueur context_length, avec stride configurable.
"""

import pickle
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class MusicDataset(Dataset):
    """
    Dataset de fenêtres glissantes pour next-token prediction.

    Chaque exemple = (input_ids, target_ids) où target = input décalé de 1.
    On concatène toutes les séquences en un seul long stream, puis on
    fenêtre. C'est plus efficace que de fenêtrer chaque morceau séparément.
    """

    def __init__(self, pkl_path, context_length=256, stride=128):
        with open(pkl_path, "rb") as f:
            sequences = pickle.load(f)

        # Concaténation : tous les morceaux à la suite, séparés par <EOS>/<BOS>
        # (déjà inclus dans chaque séquence par le tokenizer)
        all_tokens = []
        for seq in sequences:
            all_tokens.extend(seq)

        # Conversion en tensor (long pour les indices)
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

        self.context_length = context_length
        self.stride = stride

        # Nombre de fenêtres possibles
        # On a besoin de context_length + 1 tokens pour faire input/target
        self.n_windows = max(0, (len(self.tokens) - context_length - 1) // stride + 1)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        # input = tokens[start:end], target = tokens[start+1:end+1]
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1:end + 1]
        return input_ids, target_ids


def make_dataloaders(train_pkl, val_pkl, context_length=256, stride=128,
                     batch_size=32, num_workers=0):
    """
    Construit les DataLoaders train et val.
    Pour l'entraînement local Windows, mets num_workers=0 (multiprocess pose
    souvent problème). Sur Colab Linux, tu peux mettre 2.
    """
    train_ds = MusicDataset(train_pkl, context_length, stride)
    val_ds   = MusicDataset(val_pkl, context_length, stride)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds, val_ds
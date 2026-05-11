<p align="center">
  <img src="https://em-content.zobj.net/source/apple/391/cat-with-wry-smile_1f63c.png" width="80"/>
</p>

<h1 align="center">Le Chat Musicien 🎹</h1>

<p align="center">
  <strong>Un GPT from scratch qui compose de la musique</strong><br>
  <em>Projet IA — Sujet n°2 · M1 Mathématiques & Applications · Université de Lille, 2026</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Paramètres-4.88M-blueviolet" alt="Params"/>
  <img src="https://img.shields.io/badge/Perplexity-7.42-success" alt="Perplexity"/>
  <img src="https://img.shields.io/badge/Dataset-10%20855%20MIDI-orange" alt="Dataset"/>
</p>

---

## 🎯 En bref

Un modèle de langage **GPT** (Transformer décodeur), implémenté entièrement from scratch en PyTorch, entraîné sur le dataset **GrandMidiPiano** (10 855 fichiers MIDI de piano classique). On convertit la musique en tokens (représentation REMI), on entraîne le modèle à prédire le prochain token, et on génère de nouvelles compositions.

```
🎹 Fichier MIDI  →  🔤 Tokens REMI  →  🧠 GPT  →  🔤 Tokens  →  🎵 Nouvelle musique
```

**Auteur** : Mazari Kaïs — M1 M&A  
**Encadrant** : N. Wicker — Laboratoire Paul Painlevé, Université de Lille

---

## 📊 Résultats

| Métrique | Valeur |
|:---|:---|
| Paramètres | **4.88 M** |
| Tokens d'entraînement | **89 M** |
| Steps | **20 500** |
| Validation loss | **2.00** |
| Perplexity | **7.42** |
| Entraînement | ~2h20 sur Tesla T4 |

<details>
<summary><strong>📈 Courbes d'entraînement</strong></summary>
<br>

Les courbes de loss et perplexité sont disponibles dans `rapport/figs/`.

Le modèle converge proprement : la validation loss descend régulièrement de 2.81 à 2.00 sur 20 500 steps, sans overfitting visible (train et val loss restent proches).

</details>

---

## 🗂️ Structure du projet

```
chat_musicien_rendu/
│
├── src/                          # Code source
│   ├── tokenizer.py              #   Tokenizer REMI (vocab = 157 tokens)
│   ├── gpt.py                    #   Modèle GPT complet (Attention, FFN, LayerNorm)
│   ├── dataset.py                #   MusicDataset PyTorch (fenêtres glissantes)
│   ├── preprocess.py             #   MIDI → tokens .pkl (multi-process)
│   ├── train.py                  #   Boucle d'entraînement (AdamW + cosine + warmup)
│   ├── generate.py               #   Génération autorégressive (temperature, top-k, top-p)
│   └── generate_batch.py         #   Génère les 5 MIDI du rendu
│
├── notebooks/                    # Exploration & figures
│   ├── 01_explore_midi.ipynb     #   Analyse du dataset MIDI
│   └── 02_train_chat_musicien.ipynb  #   Notebook Colab d'entraînement
│
├── outputs/
│   ├── generated/                #   5 fichiers MIDI générés
│   │   ├── 01_creation_libre.mid
│   │   ├── 02_temperature_basse.mid
│   │   ├── 03_temperature_haute.mid
│   │   ├── 04_continuation_morceau_1.mid
│   │   └── 05_continuation_morceau_2.mid
│   └── models/                   #   Poids du modèle + log
│       ├── chat_musicien_best.pth
│       └── training_log.json
│
├── rapport/figs/                 #   Rapport PDF + courbes
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture

Le modèle suit l'architecture **GPT-2** (Transformer décodeur), implémentée from scratch sans utiliser de module pré-construit :

```
Token Embedding + Positional Embedding
              ↓
          Dropout
              ↓
    ┌─────────────────────┐
    │   Transformer Block  │  × 6
    │                     │
    │  LayerNorm → MHA    │  (Multi-Head Attention causale)
    │  + skip connection  │
    │                     │
    │  LayerNorm → FFN    │  (expansion 4×, GELU)
    │  + skip connection  │
    └─────────────────────┘
              ↓
        LayerNorm finale
              ↓
     Linear → logits (157)
```

| Hyperparamètre | Valeur |
|:---|:---|
| `vocab_size` | 157 (tokens REMI) |
| `context_length` | 256 |
| `emb_dim` | 256 |
| `n_heads` | 8 |
| `n_layers` | 6 |
| `drop_rate` | 0.1 |
| Activation | GELU |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Scheduler | Warmup linéaire + cosine decay |

---

## 🎵 Tokenisation REMI

Chaque note MIDI est convertie en **5 tokens** structurés :

| Token | Rôle | Exemple |
|:---|:---|:---|
| `BAR` | Début de mesure (4/4) | — |
| `POSITION_i` | Position dans la mesure (0–15, résolution double-croche) | `POSITION_4` → 2ème temps |
| `NOTE_ON_p` | Pitch MIDI (21–108, range complet du piano) | `NOTE_ON_60` → Do central |
| `DURATION_d` | Durée en steps (1–16) | `DURATION_4` → noire |
| `VELOCITY_v` | Dynamique quantifiée (0–31) | `VELOCITY_20` → forte |

**Tokens spéciaux** : `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`  
**Taille du vocabulaire** : **157 tokens** (déterministe, pas besoin de fit sur les données)

---

## 🎼 Les 5 MIDI générés

| # | Fichier | Stratégie | Temperature | Sampling |
|:--|:---|:---|:---:|:---|
| 1 | `01_creation_libre.mid` | Génération libre depuis `<BOS>` | 1.0 | top-k=30 |
| 2 | `02_temperature_basse.mid` | Conservateur, peu de risques | 0.7 | top-k=15 |
| 3 | `03_temperature_haute.mid` | Créatif, nucleus sampling | 1.2 | top-p=0.95 |
| 4 | `04_continuation_morceau_1.mid` | Continuation d'un vrai morceau | 1.0 | top-k=30 |
| 5 | `05_continuation_morceau_2.mid` | Continuation d'un autre morceau | 0.9 | top-k=25 |

> 💡 Pour écouter les fichiers `.mid`, ouvrir dans un lecteur MIDI (MuseScore, Timidity, ou [signal.vercel.app/edit](https://signal.vercel.app/edit) en ligne).

---

## 🚀 Reproduction

### 1. Installation

```bash
git clone https://github.com/<ton-username>/chat_musicien.git
cd chat_musicien
conda create -n chat_musicien python=3.11 -y
conda activate chat_musicien
pip install -r requirements.txt
```

### 2. Données

Télécharger `GrandMidiPiano.zip` et dézipper dans `data/GrandMidiPiano/`.

### 3. Prétraitement

```bash
python -m src.preprocess --n_workers 8
```

Produit `outputs/tokens/{train,val}_tokens.pkl` (~89M tokens).

### 4. Entraînement

```bash
# Sur GPU (recommandé : Google Colab Tesla T4)
python -m src.train --device cuda --epochs 2 --batch_size 64

# Test rapide CPU
python -m src.train --steps 100
```

Durée : **~2h20 sur T4**. Modèle sauvegardé dans `outputs/models/chat_musicien_best.pth`.

### 5. Génération

```bash
# Les 5 MIDI du rendu
python -m src.generate_batch --model outputs/models/chat_musicien_best.pth --seed 42

# Génération libre personnalisée
python -m src.generate \
    --model outputs/models/chat_musicien_best.pth \
    --temperature 1.0 --top_k 30 --max_tokens 800

# Continuation d'un morceau existant
python -m src.generate \
    --model outputs/models/chat_musicien_best.pth \
    --prompt_midi mon_morceau.mid \
    --temperature 0.9 --top_k 25
```

---

## 📚 Références

- Vaswani et al. — *Attention Is All You Need* (2017)
- Radford et al. — *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Huang & Yang — *Pop Music Transformer* (REMI tokenization, 2020)
- Cours de N. Wicker — *Reinforcement Learning & LLM*, Université de Lille

---

<p align="center">
  <em>Projet réalisé dans le cadre du cours d'IA de N. Wicker — Université de Lille, 2026</em>
</p>

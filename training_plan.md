# HessGPT — Plan d'entraînement complet

## Architecture finale
- embed_dim = 1120 | heads = 16 | kv = 4 | layers = 32
- head_dim = 70 | ffn_dim = 3008 | tie_embeddings = True
- **Total : 567.2M params**
- Vocab : 128 010 (Llama-3 + 8 tokens spéciaux)

---

## Vue d'ensemble

| Phase | Tokens | seq_len | YaRN | Objectif |
|-------|--------|---------|------|----------|
| 1     | 10B    | 512     | OFF  | Base langue générale |
| 2A    | 10B    | 512     | OFF  | Introduction math/code doux |
| 2B    | 8B     | 1024    | ×2   | Math + code dominant |
| 2C    | 2B     | 4096    | ×8   | Annealing contexte long |
| SFT   | ~500k ex | 4096  | ×8   | Instruction following + CoT |
| GRPO  | rewards  | 4096  | ×8   | Raisonnement math/code vérifié |

**Total pretrain : 30B tokens**

---

## Phase 1 — 10B tokens, seq=512

**Données** : mix texte général Phase 1 (déjà téléchargé)
- DCLM 37% · FineWeb-Edu 22% · Zyda-2 18% · Cosmopedia 10% (filtré math)
- peS2o 5% · FineWiki 5% · FineMath 3% · beyond_web ~100M

**Config entraînement**
```
max_seq_len           512
batch_size            150
gradient_accumulation 8
batch effectif        ~76 800 tokens/step
learning_rate         4e-4
warmup_ratio          0.03
decay_ratio           0.15
min_lr_ratio          0.1
use_yarn              False
num_chunks            5  (5 × 2.1B = 10.5B)
```

**Objectif** : construire les représentations de base — grammaire, syntaxe,
connaissances générales, structure du langage. La loss doit descendre de ~10
à ~2.8-3.2 sur ce chunk.

---

## Phase 2A — 10B tokens, seq=512

**Données** : datasets_phase2A.py
- Cosmopedia v2 25% **(sans filtre math)** · Zyda-2 20% · FineWeb-Edu 18%
- FineMath 4+ 15% · DCLM 12% · peS2o 5% · niche ~100M

**Config entraînement**
```
max_seq_len           512
batch_size            150
gradient_accumulation 8
learning_rate         2e-4   ← divisé par 2 vs Phase 1
warmup_ratio          0.01   ← court, modèle déjà stable
decay_ratio           0.15
use_yarn              False
num_chunks            5
checkpoint_file       ./Model/HessGpt_phase2a.pt
```

**Objectif** : introduction douce aux équations et au raisonnement structuré.
Cosmopedia sans filtre expose le modèle à du contenu pédagogique avec maths
sans que ce soit du raisonnement pur — transition naturelle.

---

## Phase 2B — 8B tokens, seq=1024

**Données** : datasets_phase2B.py (8 chunks)
- FineMath 4+ 30% · The Stack Python 15% · Jupyter 10%
- Cosmopedia v2 18% · peS2o 10% · DCLM 7% · FineWeb-Edu 5% · niche ~100M

**Config entraînement**
```
max_seq_len           1024
batch_size            75     ← 150/2 (seq ×2)
gradient_accumulation 16     ← 8×2 (batch effectif identique)
learning_rate         1e-4
warmup_ratio          0.01
decay_ratio           0.20
use_yarn              True
yarn_scale            2.0    ← 512 × 2 = 1024
yarn_original_max_len 512
num_chunks            4      ← 4 × 2.1B = 8.4B
checkpoint_file       ./Model/HessGpt_phase2b.pt
```

**Objectif** : math + code dominant, raisonnement long. Les notebooks Jupyter
sont de la CoT implicite (code + commentaires entrelacés). Le modèle commence
à voir des séquences qui dépassent 512 tokens.

---

## Phase 2C — 2B tokens, seq=4096 (Annealing)

**Données** : mêmes datasets que Phase 2B (1 chunk)

**Config entraînement**
```
max_seq_len           4096
batch_size            10     ← VRAM ×4 vs seq=1024
gradient_accumulation 80     ← batch effectif identique ~76 800
learning_rate         5e-5   ← très bas, annealing doux
warmup_ratio          0.005
decay_ratio           0.20
use_yarn              True
yarn_scale            8.0    ← 512 × 8 = 4096
yarn_original_max_len 512
num_chunks            1
checkpoint_file       ./Model/HessGpt_annealing.pt
```

**Objectif** : exposer les embeddings positionnels aux positions 1024-4096
avant le SFT. Sans cette étape, le premier batch SFT verrait des positions
inconnues → spike de loss. Après ce chunk le modèle connaît le long contexte.

> ⚠️ Tester batch_size=10 seq=4096 avant de lancer — si OOM descendre à
> batch_size=6 et monter gradient_accumulation=130.

---

## Phase 3 — SFT, seq=4096

**Données** (~500k exemples)
- OpenThoughts3-1.2M (traces QwQ-32B) — raisonnement math/code
- Instruction following général
- Exemples sandbox Python avec `<|code|>` / `<|result|>` / `<|error|>`

**Format séquence**
```
<|start_header_id|>system<|end_header_id|>
Tu es HessGPT...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Résous x²+5x+6=0<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
<|think|>
  je dois factoriser... discriminant = 1...
<|/think|>
<|code|>
import sympy; x=sympy.Symbol('x')
print(sympy.solve(x**2+5*x+6))
<|/code|>
<|result|>[-3, -2]<|/result|>
Les solutions sont x=-2 et x=-3<|eot_id|>
```

**Config entraînement**
```
max_seq_len           4096
batch_size            8-12
gradient_accumulation 64
learning_rate         2e-5
warmup_ratio          0.03
num_epochs            2-3
use_yarn              True
yarn_scale            8.0
loss_mask             True   ← -100 sur tokens system/user
checkpoint_file       ./Model/HessGpt_sft.pt
```

**Loss mask** : calculée uniquement sur les tokens après
`<|end_header_id|>` du tour assistant — le modèle apprend à répondre,
pas à répéter la question.

---

## Phase 4 — GRPO / RLVR

**Objectif** : affiner le raisonnement math/code via reward vérifiable

**Reward functions**
- Math : comparaison numérique de la réponse finale (exact match)
- Code : exécution sandbox → résultat correct = reward +1
- Format : présence correcte de `<|think|>` → `<|code|>` → `<|result|>`

**Config**
```
max_seq_len           4096
learning_rate         5e-6
use_yarn              True
yarn_scale            8.0
num_generations       8      ← 8 réponses par question pour GRPO
checkpoint_file       ./Model/HessGpt_grpo.pt
```

---

## Tokens spéciaux (rappel)

| Token | ID (auto) | Usage |
|-------|-----------|-------|
| `<\|think\|>` | slot libre | début raisonnement |
| `<\|/think\|>` | slot libre | fin raisonnement |
| `<\|code\|>` | slot libre | bloc code Python |
| `<\|/code\|>` | slot libre | fin bloc code |
| `<\|result\|>` | slot libre | résultat sandbox |
| `<\|/result\|>` | slot libre | fin résultat |
| `<\|error\|>` | slot libre | erreur sandbox |
| `<\|/error\|>` | slot libre | fin erreur |

IDs assignés automatiquement par `add_special_tokens()` —
jamais hardcodés.

---

## Checklist avant chaque phase

- [ ] Vérifier `data_dir` pointe vers le bon répertoire
- [ ] Vérifier `checkpoint_file` pointe vers le bon fichier
- [ ] Adapter `batch_size` + `gradient_accumulation` selon seq_len
- [ ] Activer `use_yarn=True` + bon `yarn_scale` à partir de Phase 2B
- [ ] Tester 1 batch avant de lancer (vérifier VRAM)
- [ ] Sauvegarder le tokenizer avec chaque checkpoint

---

## Récapitulatif batch effectif (constant)

| Phase | batch_size | grad_accum | seq_len | tokens/step |
|-------|-----------|------------|---------|-------------|
| 1 + 2A | 150     | 8          | 512     | 76 800      |
| 2B     | 75      | 16         | 1024    | 76 800      |
| 2C     | 10      | 80         | 4096    | 81 920 ≈    |
| SFT    | 8-12    | 64         | 4096    | ~65 536     |
| GRPO   | 4-8     | variable   | 4096    | —           |

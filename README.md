# mini_decision

Mini modèle (≈1M paramètres, entraîné from scratch) qui prend un **contexte**, une
**description du vrai** et une **description du faux**, et renvoie **P(vrai)** calibrée
(≈ 0.5 si le contexte est hors sujet).

```
Contexte : « Le client a réglé sa facture le 3 mars. »
Vrai     : « La facture est payée, le client ne doit plus rien. »
Faux     : « Le paiement est manquant ou en retard, il reste un solde dû. »
→ P(vrai) ≈ 0.95
```

Domaine des données : multi-domaine administratif en français (facturation, colis, support
client, RH, contrats), réglable dans `configs/base.yaml` (`data.domain`, `data.subdomains`).

## Installation

```bash
uv sync                      # crée .venv avec toutes les dépendances
uv run pytest                # vérifie l'installation
```

Le calcul choisit automatiquement `cuda`, puis `mps` (Apple Silicon), puis `cpu`
(`device:` dans la config).

## LLM (génération et juges)

| Rôle       | Par défaut                       | Clé / prérequis                        |
|------------|----------------------------------|----------------------------------------|
| Générateur | Anthropic `claude-haiku-5-5`     | `ANTHROPIC_API_KEY`                    |
| Juge 1     | Anthropic `claude-haiku-5-5`     | `ANTHROPIC_API_KEY`                    |
| Juge 2     | 2ᵉ tirage `claude-haiku-5-5`     | `ANTHROPIC_API_KEY` (variante Gemma local : `configs/judges_gemma.yaml`) |

Un client OpenAI est aussi disponible (`provider: openai`, clé `OPENAI_API_KEY`).

- **Cache** : chaque appel est stocké dans `.cache/llm/` (clé = sha256 du fournisseur, du
  modèle, des prompts, du schéma et du numéro de tirage). Relancer une commande ne repaie
  jamais un appel.
- **Plafond de dépense** : `llm.budget_usd` (10 $ par défaut), cumulé sur tous les runs dans
  `.cache/llm/spend.json`. Une fois atteint, les nouveaux appels payants lèvent
  `BudgetExceeded` ; le cache et les modèles locaux restent utilisables.

## Configs

`configs/base.yaml` contient toutes les valeurs ; chaque variante hérite (`inherit:`) et ne
change que ce qui diffère :

| Fichier                     | Variante                                  |
|-----------------------------|-------------------------------------------|
| `base.yaml`                 | transformer + tête QDA diagonale          |
| `ngram.yaml`, `cnn.yaml`    | autres encodeurs                          |
| `transformer_dot.yaml`      | tête produit scalaire (baseline)          |
| `transformer_full.yaml`     | covariance pleine                         |
| `transformer_no_mlm.yaml`   | sans pré-entraînement MLM                 |
| `transformer_aux_mlm.yaml`  | loss auxiliaire MLM (λ = 0.1)             |
| `debug.yaml`                | run minuscule pour tester le pipeline     |

Surcharge ponctuelle : `--set model.head=dot --set train.lr=5e-4`.

## Pipeline

Chaque étape lit `--config` (par défaut `configs/base.yaml`).

```bash
uv run mini-decision datagen            # paires + contextes + corpus non étiqueté   (jalon 2)
uv run mini-decision verify             # labels souples par 2 juges, rejet si écart > 0.3
uv run mini-decision split              # train/val/test par pair_id + contrôle de fuite
uv run mini-decision train-tokenizer    # SentencePiece unigram, vocab 4000           (jalon 3)
uv run mini-decision pretrain-mlm       # MLM from scratch                            (jalon 4)
uv run mini-decision train              # supervisé, BCE sur labels souples
uv run mini-decision calibrate          # temperature scaling                         (jalon 5)
uv run mini-decision evaluate           # métriques + diagramme de fiabilité
uv run mini-decision report             # rapport Markdown comparatif
uv run mini-decision adversarial-round  # une ronde model-in-the-loop                 (jalon 6)
```

État : jalons 1 et 2 faits (`datagen`, `verify`, `split`) ; les autres commandes sont vides.

### Données (jalon 2)

```bash
uv run mini-decision datagen   # data/raw/pairs.jsonl + data/raw/unlabeled.jsonl
uv run mini-decision verify    # data/processed/labeled.jsonl, rejected.jsonl, judge_spreads.json
uv run mini-decision split     # data/processed/{train,val,test,test_counterfactual}.jsonl + stats.md
```

- `datagen --skip-unlabeled` ne génère que les paires.
- `verify` rejette un contexte si les juges divergent de plus de `judge_max_disagreement`,
  si l'un d'eux le juge hors sujet alors qu'il ne devait pas l'être (ou l'inverse), s'il
  est trop long, ou si le label moyen contredit le kind demandé au générateur.
- `split` lance aussi deux baselines (régression logistique sur n-grammes de caractères) :
  *descriptions seules* (contrôle de fuite de la spec, doit rester vers 50 %) et *contexte
  seul* (détecte les artefacts de style, par exemple « la négation indique toujours le faux »).
- Toutes les statistiques et 10 exemples par kind sont dans `data/processed/stats.md`.
- Les commandes sont relançables : les appels déjà faits viennent du cache.

## Structure

```
configs/               base.yaml + variantes
data/                  raw/, processed/, adversarial/round_k/, real_calibration.jsonl (à fournir)
src/mini_decision/
  config.py            chargement YAML typé (pydantic), héritage, surcharges
  schemas.py           Example (format JSONL) + sorties structurées des LLM
  utils.py             graines, choix du device
  llm/                 LLMClient, Anthropic / OpenAI / Ollama, cache disque, plafond, prompts
  datagen/ tokenizer/ model/ train/ adversarial/ eval/
  cli.py
tests/
```

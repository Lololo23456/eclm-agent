# /train

Lance l'entraînement d'un composant spécifique.

## Usage
```
/train intent          ← Fine-tune CamemBERT Intent Extractor
/train eclm            ← Entraîne l'ECLM Core
/train planner         ← Entraîne l'AST Planner
/train testgen         ← Entraîne le TestGenerator
/train dpo             ← Lance le DPO mensuel sur paires collectées
```

## Ce que tu dois faire pour chaque composant

### /train intent
1. Vérifier que `data/training/intent/intent_train.jsonl` existe et contient ≥ 500 exemples
2. Lancer :
   ```bash
   python src/intent/train.py \
     --data data/training/intent/ \
     --output models/intent/ \
     --epochs 5 \
     --batch_size 16 \
     --lr 2e-5
   ```
3. Évaluer sur `intent_val.jsonl` — afficher accuracy, F1 par classe d'action
4. Si accuracy < 95% → analyser les classes faibles et proposer des exemples supplémentaires

### /train eclm
1. Vérifier que le curriculum est trié par complexité (`data/training/eclm/`)
2. Vérifier que le Docker sandbox est disponible (`docker ps`)
3. Lancer en mode curriculum :
   ```bash
   python src/eclm/train.py \
     --curriculum data/training/eclm/ \
     --output models/eclm/ \
     --reward execution \
     --beam_width 5 \
     --sandbox docker
   ```
4. Logger les reward moyens par epoch — alerter si plateau > 3 epochs

### /train dpo
1. Compter les paires dans `data/dpo_pairs/` — minimum 100 paires pour un run utile
2. ```bash
   python scripts/monthly_finetune.py \
     --pairs data/dpo_pairs/ \
     --base_model models/eclm/ \
     --output models/eclm_dpo/
   ```
3. Lancer `/benchmark` automatiquement après
4. Déployer `models/eclm_dpo/` seulement si benchmark_score > benchmark_score_actuel

### /train planner
```bash
python src/planner/train.py \
  --data data/training/planner/ \
  --output models/planner/ \
  --epochs 10
```

### /train testgen
```bash
python src/verifier/test_generator/train.py \
  --data data/training/test_generator/ \
  --output models/testgen/ \
  --epochs 8
```

## Règles
- JAMAIS déployer un modèle sans avoir lancé `/benchmark` d'abord
- JAMAIS entraîner l'ECLM sur des données non vérifiées par le Verifier
- Toujours sauvegarder le modèle précédent avant d'écraser (backup automatique)

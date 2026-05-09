# /dpo-collect

Collecte manuellement une paire DPO ou affiche les statistiques de collecte.

## Usage
```
/dpo-collect status           ← combien de paires, répartition par mois
/dpo-collect manual           ← ajouter une correction manuelle
/dpo-collect export           ← exporter toutes les paires en JSONL propre
```

## Ce que tu dois faire

### /dpo-collect status
```python
from src.improvement.dpo_collector import DPOCollector
from src.shared.config import Config

collector = DPOCollector(Config())
print(f"Total paires DPO : {collector.count()}")
```
Rappel des seuils :
- 100 paires → `/train dpo` devient disponible
- 500 paires → DPO mensuel devient stable

### /dpo-collect manual
Workflow interactif :
1. Demander : "Quelle était la commande ?"
2. Demander : "Quel code l'agent a-t-il généré (rejeté) ?"
3. Demander : "Quel est le code correct (choisi) ?"
4. Appeler `DPOCollector.collect_manual(prompt, chosen, rejected)`
5. Confirmer l'enregistrement

### /dpo-collect export
Fusionner tous les `dpo_*.jsonl` en un seul fichier propre avec déduplication :
```python
import hashlib, json
from src.shared.config import Config
config = Config()
seen = set()
with open(config.dpo_pairs_dir / "dpo_all_clean.jsonl", "w") as out:
    for path in sorted(config.dpo_pairs_dir.glob("dpo_*.jsonl")):
        for line in open(path):
            h = hashlib.sha256(line.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                out.write(line)
```

## Règles
- chosen_score DOIT être ≥ 0.8 (validé par le verifier)
- chosen_score DOIT être > rejected_score
- Source "manual_correction" = correction humaine → prioritaire en DPO

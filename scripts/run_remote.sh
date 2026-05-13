#!/bin/bash
# Envoie un plan.json vers le serveur ECLM distant et lance l'exécution.
# Usage: ./scripts/run_remote.sh data/plans/mon_plan.json http://192.168.1.x:8765
set -e

PLAN_FILE="${1:-}"
SERVER_URL="${2:-http://localhost:8765}"

if [ -z "$PLAN_FILE" ]; then
    echo "Usage: $0 <plan.json> [server_url]"
    echo "  Exemple: $0 data/plans/api_rest.json http://192.168.1.42:8765"
    exit 1
fi

if [ ! -f "$PLAN_FILE" ]; then
    echo "Plan introuvable: $PLAN_FILE"
    exit 1
fi

echo "Envoi de $PLAN_FILE vers $SERVER_URL..."

# Envelopper le plan dans {"plan": ...}
PAYLOAD=$(python3 -c "
import json, sys
plan = json.loads(open('$PLAN_FILE').read())
print(json.dumps({'plan': plan}))
")

curl -s -X POST "$SERVER_URL/v1/pipeline/run" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" | python3 -m json.tool

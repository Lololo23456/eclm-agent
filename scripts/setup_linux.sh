#!/bin/bash
# Setup ECLM Agent sur Ubuntu 24.04 LTS
# Usage: bash scripts/setup_linux.sh
set -e

echo "=== ECLM Agent — Setup Ubuntu 24.04 ==="

# 1. Python 3.11 + pip
echo "[1/6] Python + pip..."
sudo apt-get update -q
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl

# 2. Ollama
echo "[2/6] Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Ollama en service systemd (démarrage auto)
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 3

# Modèles
echo "  → Pull qwen2.5-coder:7b (~4.7 GB)..."
ollama pull qwen2.5-coder:7b

# phi3:mini pour IntegratorAgent (3.8B, rapide)
echo "  → Pull phi3:mini (~2.3 GB)..."
ollama pull phi3:mini

# 3. Cloner le projet
echo "[3/6] ECLM Agent..."
cd ~
if [ ! -d "eclm-agent" ]; then
    git clone git@github.com:Lololo23456/eclm-agent.git
fi
cd eclm-agent

# 4. Environnement Python
echo "[4/6] Environnement Python..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 5. .env
echo "[5/6] Configuration..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Ollama — sur ce serveur
OLLAMA_BASE_URL=http://localhost:11434
ECLM_FAST_MODEL=qwen2.5-coder:7b
ECLM_STRONG_MODEL=qwen2.5-coder:7b

# Claude API (optionnel, pour /plan depuis MacBook)
# ANTHROPIC_API_KEY=sk-...
# ECLM_USE_CLAUDE_API=true

# Performance
ECLM_MAX_PARALLEL_TASKS=4
ECLM_LOCAL_SANDBOX=true
EOF
    echo "  → .env créé (édite si besoin)"
fi

# 6. Service systemd pour l'API
echo "[6/6] Service ECLM API..."
sudo tee /etc/systemd/system/eclm-api.service > /dev/null << EOF
[Unit]
Description=ECLM Agent API Server
After=ollama.service
Requires=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/eclm-agent
ExecStart=$HOME/eclm-agent/.venv/bin/python -m src.api.server --host 0.0.0.0 --port 8765
Restart=always
RestartSec=5
EnvironmentFile=$HOME/eclm-agent/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable eclm-api
sudo systemctl start eclm-api

# Test rapide
sleep 2
echo ""
echo "=== Vérification ==="
curl -s http://localhost:8765/health | python3 -m json.tool || echo "  API pas encore prête (attendre 5s)"

echo ""
echo "=== Setup terminé ==="
echo ""
echo "  API locale   : http://localhost:8765"
echo "  API distant  : http://$(hostname -I | awk '{print $1}'):8765"
echo ""
echo "Depuis le MacBook :"
echo "  # Envoyer un plan et exécuter sur ce serveur :"
echo "  ./scripts/run_remote.sh data/plans/mon_plan.json http://$(hostname -I | awk '{print $1}'):8765"
echo ""
echo "  # Générer un plan sans tokens :"
echo "  curl -X POST http://$(hostname -I | awk '{print $1}'):8765/v1/pipeline/plan \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"brief\": \"crée une API REST simple\"}'"

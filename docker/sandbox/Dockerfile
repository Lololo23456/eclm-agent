FROM python:3.11-slim

# Aucun accès réseau pendant l'exécution (--network none au runtime)
WORKDIR /app

# Outils de vérification
RUN pip install --no-cache-dir \
    mypy==1.9.0 \
    ruff==0.3.0 \
    pytest==8.0.0 \
    hypothesis==6.100.0 \
    pytest-cov==5.0.0

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 sandbox
USER sandbox

# Timeout par défaut (override au runtime)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "print('ECLM Sandbox ready')"]

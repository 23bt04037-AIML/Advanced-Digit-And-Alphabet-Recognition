#!/bin/bash
# entrypoint.sh – runs inside the container before supervisor starts
set -e

echo "=== DigitAI Startup ==="

# Seed DB if it doesn't already have model_metrics rows
python -c "
from backend.database.db import engine, Base
Base.metadata.create_all(bind=engine)
from backend.database.db import SessionLocal
from backend.database.models import ModelMetrics
db = SessionLocal()
count = db.query(ModelMetrics).count()
db.close()
if count == 0:
    import subprocess, sys
    subprocess.run([sys.executable, 'seed_db.py'], check=False)
    print('DB seeded.')
else:
    print(f'DB already has {count} model_metrics rows – skipping seed.')
"

echo "=== Starting services via supervisord ==="
exec supervisord -c /etc/supervisor/conf.d/digitai.conf

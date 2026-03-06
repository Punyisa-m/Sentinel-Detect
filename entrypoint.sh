#!/bin/sh
# entrypoint.sh — init DB + model before starting the main process

set -e

echo "🔧 Checking database..."
python - <<'EOF'
import sys, os
sys.path.insert(0, '/app')
from pathlib import Path
from data_generator import init_db, generate_normal_log, inject_anomalies, engineer_features, store_logs, DB_PATH

db = Path(DB_PATH)

# Init tables always (safe to run multiple times)
init_db(DB_PATH)

# Seed data only if table is empty
import sqlite3
conn = sqlite3.connect(DB_PATH)
count = conn.execute("SELECT COUNT(*) FROM activity_logs").fetchone()[0]
conn.close()

if count == 0:
    print("📊 No data found — generating synthetic logs...")
    normal = generate_normal_log(500)
    full   = inject_anomalies(normal, 40)
    feat   = engineer_features(full)
    store_logs(feat, DB_PATH)
    print(f"✅ {len(feat)} records inserted")
else:
    print(f"✅ DB already has {count} records — skipping seed")
EOF

echo "🤖 Checking model..."
python - <<'EOF'
import sys
sys.path.insert(0, '/app')
from pathlib import Path
from model_engine import train_model, score_logs, MODEL_PATH
from data_generator import DB_PATH

if not Path(MODEL_PATH).exists():
    print("Training Isolation Forest model...")
    train_model(DB_PATH)
    score_logs(DB_PATH, retrain=False)
    print("✅ Model ready")
else:
    print("✅ Model already exists — skipping training")
EOF

echo "🚀 Starting application..."
exec "$@"
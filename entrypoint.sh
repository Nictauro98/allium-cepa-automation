#!/bin/bash
set -e

# Start API in background; Streamlit is the foreground process HF Spaces monitors.
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for API to become healthy..."
for _ in $(seq 1 30); do
    curl -sf http://localhost:8000/health > /dev/null 2>&1 && { echo "API ready."; break; }
    sleep 2
done

exec streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true

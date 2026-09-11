#!/bin/bash
set -e

# Run Streamlit on port 3000 for AI Studio reverse proxy
exec python3 -m streamlit run app.py \
  --server.port 3000 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false

# start_mcp.sh
#!/bin/bash
cd # input project directory here
docker compose up -d postgres
sleep 3  # wait for postgres to be ready
source .venv/bin/activate
python mcp_server.py

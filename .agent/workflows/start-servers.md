---
description: Start the backend and frontend servers for the Delorean Strategy Dashboard
---

# Start Servers Workflow

This workflow starts both the FastAPI backend and Next.js frontend servers.

## Prerequisites
- Ensure you are in the `quant` conda environment
- Backend dependencies installed (`pip install -r requirements.txt`)
- Frontend dependencies installed (`cd frontend && npm install`)

## Steps

// turbo-all

1. Start the backend server (FastAPI on port 8000):
```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null || true && cd /Users/jinjing/workspace/delorean && conda run -n quant python server/main.py
```

2. Start the frontend server (Next.js on port 3000):
```bash
lsof -ti:3000 | xargs kill -9 2>/dev/null || true && cd /Users/jinjing/workspace/delorean/frontend && npm run dev
```

## Access Points
- **Dashboard UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## Notes
- Both servers should run in separate terminals
- The backend must be running for the frontend to fetch data
- To stop either server, use `Ctrl+C` in the respective terminal

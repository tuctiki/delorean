"""
Delorean Strategy Dashboard - FastAPI Backend

This is the main entry point for the API server.
Routes are organized into modular files under server/routes/.
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize Qlib
try:
    import qlib
    from delorean.config import QLIB_PROVIDER_URI, QLIB_REGION
    qlib.init(provider_uri=QLIB_PROVIDER_URI, region=QLIB_REGION)
    print(f"Qlib initialized with {QLIB_PROVIDER_URI}")
except ImportError:
    print("Warning: Qlib not found or configuration error. Detailed data features will be disabled.")
except Exception as e:
    print(f"Warning: Qlib init failed: {e}")

# Create FastAPI app
app = FastAPI(title="Delorean Strategy Dashboard")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include modular routers
from server.routes.experiments import router as experiments_router
from server.routes.tasks import router as tasks_router
from server.routes.data import router as data_router

app.include_router(experiments_router)
app.include_router(tasks_router)
app.include_router(data_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

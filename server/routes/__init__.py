"""
Server routes package.
Contains modular route definitions for the Delorean API.
"""

from server.routes.experiments import router as experiments_router
from server.routes.tasks import router as tasks_router
from server.routes.data import router as data_router

__all__ = ["experiments_router", "tasks_router", "data_router"]

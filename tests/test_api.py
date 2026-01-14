"""
API integration tests for the Delorean Strategy Dashboard backend.
Tests the key FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.main import app

client = TestClient(app)


class TestStatusEndpoint:
    """Tests for the /api/status endpoint."""
    
    def test_status_returns_json(self):
        """Test that status endpoint returns valid JSON."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
        assert "log" in data
    
    def test_status_running_is_boolean(self):
        """Test that running field is a boolean."""
        response = client.get("/api/status")
        data = response.json()
        assert isinstance(data["running"], bool)


class TestExperimentsEndpoint:
    """Tests for the /api/experiments endpoint."""
    
    def test_experiments_returns_list(self):
        """Test that experiments endpoint returns a list."""
        response = client.get("/api/experiments")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_experiments_run_structure(self):
        """Test that experiment runs have required fields."""
        response = client.get("/api/experiments")
        data = response.json()
        
        if len(data) > 0 and data[0].get("id") != "error":
            run = data[0]
            assert "id" in run
            assert "name" in run
            assert "params" in run
            assert "metrics" in run


class TestRecommendationsEndpoint:
    """Tests for the /api/recommendations endpoint."""
    
    def test_recommendations_returns_json(self):
        """Test that recommendations endpoint returns valid JSON."""
        response = client.get("/api/recommendations")
        assert response.status_code == 200
        # Returns dict (possibly empty)
        data = response.json()
        assert isinstance(data, dict)


class TestConfigEndpoint:
    """Tests for the /api/config endpoint."""
    
    def test_config_returns_json(self):
        """Test that config endpoint returns valid JSON."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
    
    def test_config_has_required_sections(self):
        """Test that config has all required sections."""
        response = client.get("/api/config")
        data = response.json()
        
        assert "model_params" in data
        assert "data_factors" in data
        assert "universe" in data
        assert "time_range" in data
    
    def test_config_universe_is_list(self):
        """Test that universe is a list of ETF symbols."""
        response = client.get("/api/config")
        data = response.json()
        
        assert isinstance(data["universe"], list)
        assert len(data["universe"]) > 0
        # Each symbol should be a string
        for symbol in data["universe"]:
            assert isinstance(symbol, str)


class TestSearchEndpoint:
    """Tests for the /api/search endpoint."""
    
    def test_search_returns_etf_list(self):
        """Test that search endpoint returns ETF list."""
        response = client.get("/api/search")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestBacktestStatusEndpoint:
    """Tests for the /api/backtest-status endpoint."""
    
    def test_backtest_status_returns_json(self):
        """Test that backtest status endpoint returns valid JSON."""
        response = client.get("/api/backtest-status")
        assert response.status_code == 200
        data = response.json()
        assert "running" in data
        assert "log" in data


class TestExperimentResultsEndpoint:
    """Tests for the /api/experiment_results endpoint."""
    
    def test_experiment_results_returns_json(self):
        """Test that experiment results endpoint returns valid JSON."""
        response = client.get("/api/experiment_results")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

"""
E2E Test leveraging Playwright to verify the Gradio tuple contract explicitly.
"""
from typing import Generator
import pytest
import subprocess
import time
import requests
import os
import sys
from playwright.sync_api import Page, expect

PORT = 17862

@pytest.fixture(scope="session", autouse=True)
def run_app():
    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(PORT)
    env["OPENROUTER_API_KEY"] = "sk-or-v1-mock-key"
    
    process = subprocess.Popen(
        [sys.executable, "run.py"], 
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    
    url = f"http://127.0.0.1:{PORT}"
    started = False
    for _ in range(60):
        try:
            if requests.get(url).status_code == 200:
                started = True
                break
        except Exception:
            time.sleep(0.5)
            
    if not started:
        process.terminate()
        _, err = process.communicate()
        raise RuntimeError(f"Server did not start at {url}. Err: {err.decode('utf-8')}")
        
    yield url
    process.terminate()
    process.wait(timeout=5)

def test_app_yield_contract(page: Page, run_app: str):
    page.goto(run_app)
    expect(page.get_by_text("OpenRouter Free Model Benchmarker")).to_be_visible(timeout=5000)

    # Click Discover to fetch models (Public endpoint, works without auth)
    page.get_by_role("button", name="Discover Free Models").click()
    
    # Wait for the real models to load
    expect(page.get_by_text("free models")).to_be_visible(timeout=10000)
    
    # Select Quick Reasoning preset
    page.locator('.wrap:has-text("Single Preset")').locator('input').click()
    page.get_by_text("Quick Reasoning", exact=False).click()

    # Expand the model dropdown (label: Select Models to Benchmark)
    page.locator('.wrap:has-text("Select Models to Benchmark")').locator('input').click()
    # Click the first available option in the rendered dropdown list
    page.locator('.options li.selected').click()
    
    # Run the benchmark (This will fail due to mock API key, but the UI must not crash)
    page.get_by_role("button", name="Run Benchmark").click()
    
    # Wait for Completion Message
    expect(page.get_by_text("Complete")).to_be_visible(timeout=20000)

    # Check Leaderboard Tab - It should render despite errors
    page.get_by_text("Leaderboard").click()
    leaderboard = page.locator("table")
    expect(leaderboard).to_be_visible()

"""
app.py — Top-level Entry Point (Backward Compatibility)
───────────────────────────────────────────────────────
This module simply exports the `build_app` function from `ui_components.py`.
State management and UI generation have been decoupled into 
`state_managers.py` and `ui_components.py` respectively.
"""

from ui_components import build_app

__all__ = ["build_app"]

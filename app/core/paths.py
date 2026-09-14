"""Filesystem paths used to wire up Jinja2 templates and static assets.

Kept in its own module (rather than inline in main.py) so both
`app/web/pages.py` and `main.py` can import the same constants without
a circular import.
"""

from pathlib import Path

# app/core/paths.py -> parent is app/core -> parent.parent is app/
APP_DIR = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

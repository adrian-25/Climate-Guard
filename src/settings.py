"""Environment configuration helpers for ClimateGuard."""

from pathlib import Path

from dotenv import load_dotenv


def load_project_environment(project_root: Path) -> bool:
    """Load the project's optional ``.env`` file without replacing real env vars.

    Hosting providers and Docker inject configuration through the process
    environment. Keeping ``override=False`` lets those production values take
    precedence while still making a local ``.env`` work with ``uvicorn app:app``.
    """

    return load_dotenv(dotenv_path=project_root / ".env", override=False)

"""Nox sessions."""

import tempfile
from typing import Any

import nox
from nox.sessions import Session

nox.options.sessions = "black","flake8","mypy"


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    session.install("black")
    session.run("black", "src")


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Run lint code formatter."""
    session.install("flake8")
    session.run("flake8", "src")

@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Run mypy code analyzer."""
    session.install("mypy")
    session.run("mypy", "src")

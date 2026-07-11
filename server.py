"""Gunicorn and local entrypoint for the visible-surface analyzer."""

from backend_app import app, main


if __name__ == "__main__":
    main()

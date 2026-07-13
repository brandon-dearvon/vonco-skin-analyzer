#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Von & Co — Skin Analyzer Launcher (macOS)                  ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Double-click this file to start the skin analyzer.
# It will install dependencies, start the server, and open your browser.

cd "$(dirname "$0")"

echo ""
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║  Von & Co — AI Skin Analyzer                                ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "  ❌ Python 3 is required. Install from https://python.org"
    echo "  Press any key to exit..."
    read -n 1
    exit 1
fi

# Create .env from env.txt if neither exists
if [ ! -f .env ] && [ ! -f env.txt ]; then
    echo "  ⚠️  No config found. Edit env.txt to add your GOOGLE_API_KEY"
    echo ""
fi

# Load port from env.txt or .env
if [ -f env.txt ]; then
    PORT=$(grep -s "^PORT=" env.txt | cut -d'=' -f2)
elif [ -f .env ]; then
    PORT=$(grep -s "^PORT=" .env | cut -d'=' -f2)
fi
PORT=${PORT:-5002}

# Install dependencies if Flask or the pinned Google SDK capability is missing.
if ! python3 -c "import flask; import importlib.metadata as m; from google.genai import types; assert m.version('google-genai') == '2.11.0'; assert types.ThinkingLevel.HIGH.value == 'HIGH'" 2>/dev/null; then
    echo "  📦 Installing dependencies..."
    if ! pip3 install -r requirements.txt; then
        echo "  ❌ Dependency installation failed. Review the error above and try again."
        exit 1
    fi
    echo ""
fi

echo "  🚀 Starting server on http://localhost:$PORT"
echo "  📱 Press Ctrl+C to stop"
echo ""

# Open browser after a short delay
(sleep 2 && open "http://localhost:$PORT") &

# Start the server with auto-restart on crash
while true; do
    python3 server.py
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        # Clean exit (Ctrl+C), stop
        break
    fi
    echo ""
    echo "  ⚠️  Server crashed (exit code $EXIT_CODE). Restarting in 2 seconds..."
    echo ""
    sleep 2
done

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
    echo "  ⚠️  No config found. Edit env.txt to add your ANTHROPIC_API_KEY"
    echo ""
fi

# Load port from env.txt or .env
if [ -f env.txt ]; then
    PORT=$(grep -s "^PORT=" env.txt | cut -d'=' -f2)
elif [ -f .env ]; then
    PORT=$(grep -s "^PORT=" .env | cut -d'=' -f2)
fi
PORT=${PORT:-5002}

# Install dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "  📦 Installing dependencies..."
    pip3 install -r requirements.txt
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

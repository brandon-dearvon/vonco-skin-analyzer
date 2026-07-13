"""Gunicorn configuration for Von & Co Skin Analyzer"""
import os

# Bind to Render's PORT
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Use gthread workers so heartbeat thread stays alive during long API calls
worker_class = "gthread"
threads = 4
workers = 1

# 120s timeout — critical for Gemini (10s) + Claude (15s) pipeline
timeout = 120

# Recycle workers every 50 requests to prevent memory leaks
max_requests = 50
max_requests_jitter = 5

# Use shared memory for heartbeat file (faster on containers)
worker_tmp_dir = "/dev/shm"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

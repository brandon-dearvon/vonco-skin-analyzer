"""Gunicorn configuration for Von & Co Skin Analyzer"""
import os

# Bind to Render's PORT
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Use gthread workers so health checks can respond during provider calls.
worker_class = "gthread"
threads = 4
workers = 1

# Three provider attempts are capped at 38 seconds each.
timeout = 120

# Recycle workers every 50 requests to prevent memory leaks
max_requests = 50
max_requests_jitter = 5

# Use shared memory for heartbeat files when the host provides it.
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None

# Logging: disable the default access log because it includes remote IP addresses.
accesslog = None
errorlog = "-"
loglevel = "info"

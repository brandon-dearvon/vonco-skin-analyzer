web: gunicorn server:app --bind 0.0.0.0:$PORT --timeout 120 --worker-class gthread --threads 4 --max-requests 50 --max-requests-jitter 5 --worker-tmp-dir /dev/shm

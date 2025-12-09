import os
import multiprocessing

bind = "0.0.0.0:" + os.environ.get("PORT", "3000")
workers = 1
worker_class = "sync"
timeout = 120

#! /usr/bin/env python3
import time, os

exe_path = "gemm/gemm"
gemm_timestamp = None
watchdog_timestamp = time.time()
timeout = 300

while time.time() - watchdog_timestamp <= timeout:
    time.sleep(0.001)
    tmp_timestamp = os.stat(exe_path).st_mtime
    if tmp_timestamp != gemm_timestamp:
        os.system(exe_path)
        gemm_timestamp = tmp_timestamp
        watchdog_timestamp = time.time()
        

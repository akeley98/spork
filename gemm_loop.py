#! /usr/bin/env python3
import sys, time, os

exe_path = "gemm/gemm"
gemm_timestamp = None
watchdog_timestamp = time.time()
timeout = 300

while time.time() - watchdog_timestamp <= timeout:
    time.sleep(0.001)
    tmp_timestamp = os.stat(exe_path).st_mtime
    if tmp_timestamp != gemm_timestamp:
        print(time.strftime("\x1b[34mSTART\x1b[0m %H:%M:%S", time.localtime()), file=sys.stderr)
        os.system(exe_path)
        print("\x1b[34mEND\x1b[0m", file=sys.stderr)
        gemm_timestamp = tmp_timestamp
        watchdog_timestamp = time.time()
        

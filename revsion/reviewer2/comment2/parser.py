import re
from datetime import datetime
import numpy as np
def parse_log(filename):
    runtimes = []
    mious = []

    after_marker = False
    prev_time = None

    # timestamp: [06/22 18:19:05 ScanNet]:
    time_pattern = re.compile(r"\[(\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})")

    # start line: Test [0]/[312] cloud...
    start_pattern = re.compile(r"Test\s+\[\d+\]/\[\d+\]")

    # result line: [0]/[312] cloud, ... test_miou: ...
    result_pattern = re.compile(
        r"\[\d+\]/\[\d+\].*?test_miou:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            if not after_marker:
                if line.startswith("first few ="):
                    after_marker = True
                continue

            time_match = time_pattern.search(line)
            if not time_match:
                continue

            date_str, time_str = time_match.groups()
            current_time = datetime.strptime(
                f"2024/{date_str} {time_str}",
                "%Y/%m/%d %H:%M:%S"
            )

            # Start line
            if start_pattern.search(line):
                prev_time = current_time
                continue

            # Result line
            result_match = result_pattern.search(line)
            if result_match:
                if prev_time is None:
                    continue

                runtime = (current_time - prev_time).total_seconds()
                prev_time = current_time

                miou = float(result_match.group(3))   # last value

                runtimes.append(runtime)
                mious.append(miou)

    return runtimes, mious

np.set_printoptions(suppress=True, precision=3)
# Example
baseline_runtimes, baseline_mious = parse_log("scannet/pointnext_baseline.txt")
proposed_runtimes, proposed_mious = parse_log("scannet/pointnext_proposed2.txt")

#print(runtimes)
#print(mious)

baseline_runtimes = np.array(baseline_runtimes)
proposed_runtimes = np.array(proposed_runtimes)

baseline_mious = np.array(baseline_mious)
proposed_mious = np.array(proposed_mious)

diff = baseline_mious - proposed_mious

# indices sorted by closeness to 3
idx = np.argsort(np.abs(diff - 3))
sorted_diff = diff[idx]

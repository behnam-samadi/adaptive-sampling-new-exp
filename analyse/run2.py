import re
from datetime import datetime

def parse_log_file(path):
    times = []
    mious = []
    time_miou_pairs = []

    # Matches lines like:
    # [06/10 04:11:38 ScanNet]: ... test_miou: 90.01 90.39 57.97
    pattern = re.compile(
        r"\[(\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+ScanNet\].*?test_miou:\s*[\d.]+\s+[\d.]+\s+([\d.]+)"
    )

    prev_time = None
    elapsed = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Remove ANSI color codes
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)

            m = pattern.search(line)
            if not m:
                continue

            timestamp_str = m.group(1)  # e.g. "06/10 04:11:38"
            miou = float(m.group(2))    # e.g. 57.97

            current_time = datetime.strptime(timestamp_str, "%m/%d %H:%M:%S")

            if prev_time is None:
                # For the first entry use 04:11:05 as requested
                first_time = datetime.strptime(
                    timestamp_str[:6] + "04:11:05",
                    "%m/%d %H:%M:%S"
                )
                delta_seconds = (current_time - first_time).total_seconds()
            else:
                delta_seconds = (current_time - prev_time).total_seconds()

            elapsed += delta_seconds

            times.append(elapsed)
            mious.append(miou)

            # Store time and mIoU together
            time_miou_pairs.append((elapsed, miou))

            prev_time = current_time

    return times, mious, time_miou_pairs


# Example usage
times, mious, time_miou_pairs = parse_log_file("baseline_scannet.txt")

print("times =", times)
print("mious =", mious)

print("\nTime / mIoU pairs:")
for t, m in time_miou_pairs:
    print(f"{t:.0f}s -> {m:.2f}")

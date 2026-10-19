import re
from pathlib import Path
from datetime import datetime


def parse_log(log_path):
    data = {
        "start_time": None,
        "end_time": None,
        "elapsed_time": None,
        "total_preprocess_time": 0.0,
        "network_time": None,
        "miou": None,
    }

    # remove ANSI escape sequences
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    time_pattern = re.compile(r"\[(\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})")
    miou_pattern = re.compile(r"so far miou is:\s*([0-9.]+)")
    preprocess_pattern = re.compile(
        r"Pre\s*PRocessing\s*time:\s*([0-9.]+)", re.IGNORECASE
    )
    test_pattern = re.compile(r"Test\s+\[(\d+)\]/\[(\d+)\]\s+cloud")

    last_miou = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = ansi_escape.sub("", raw_line)

            # -----------------------------
            # start time
            # -----------------------------
            if "length of test dataset" in line:
                m = time_pattern.search(line)
                if m:
                    data["start_time"] = m.group(2)

            # -----------------------------
            # mIoU
            # -----------------------------
            m = miou_pattern.search(line)
            if m:
                last_miou = float(m.group(1))

            # -----------------------------
            # preprocessing time
            # -----------------------------
            m = preprocess_pattern.search(line)
            if m:
                data["total_preprocess_time"] += float(m.group(1))

            # -----------------------------
            # Test lines
            # -----------------------------
            m = test_pattern.search(line)
            if m:
                current_test = int(m.group(1))

                # We want the mIoU reported BEFORE Test[14]
                if current_test == 14:
                    t = time_pattern.search(line)
                    if t:
                        data["end_time"] = t.group(2)
                    data["miou"] = last_miou

    # ---------------------------------
    # elapsed time
    # ---------------------------------
    if data["start_time"] and data["end_time"]:
        t0 = datetime.strptime(data["start_time"], "%H:%M:%S")
        t1 = datetime.strptime(data["end_time"], "%H:%M:%S")

        elapsed = (t1 - t0).total_seconds()
        if elapsed < 0:  # crossed midnight
            elapsed += 24 * 3600

        data["elapsed_time"] = elapsed
        data["network_time"] = (
            elapsed - data["total_preprocess_time"]
        )

    return data

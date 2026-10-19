import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


############################################################
# Parse one log file
############################################################

def parse_log(log_path):
    data = {
        "start_time": None,
        "end_time": None,
        "elapsed_time": None,
        "total_preprocess_time": 0.0,
        "network_time": None,
        "miou": None,
    }

    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    time_pattern = re.compile(r"\[(\d{2}/\d{2}) (\d{2}:\d{2}:\d{2})")
    miou_pattern = re.compile(r"so far miou is:\s*([0-9.]+)")
    preprocess_pattern = re.compile(
        r"Pre\s*PRocessing\s*time:\s*([0-9.]+)",
        re.IGNORECASE,
    )
    test_pattern = re.compile(r"Test\s+\[(\d+)\]/\[(\d+)\]\s+cloud")

    last_miou = None
    finished = False

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = ansi_escape.sub("", raw_line)

                # Start time
                if "length of test dataset" in line:
                    m = time_pattern.search(line)
                    if m:
                        data["start_time"] = m.group(2)

                # mIoU
                m = miou_pattern.search(line)
                if m:
                    last_miou = float(m.group(1))

                # preprocessing time
                m = preprocess_pattern.search(line)
                if m:
                    data["total_preprocess_time"] += float(m.group(1))

                # Test lines
                m = test_pattern.search(line)
                if m:
                    current_test = int(m.group(1))

                    # Your logs end at Test [14]
                    if current_test == 14:
                        t = time_pattern.search(line)
                        if t:
                            data["end_time"] = t.group(2)

                        data["miou"] = last_miou
                        finished = True

    except Exception as e:
        print(f"Could not read {log_path.name}: {e}")
        return None

    # Log is still being written
    if not finished:
        return None

    # Compute elapsed time
    try:
        t0 = datetime.strptime(data["start_time"], "%H:%M:%S")
        t1 = datetime.strptime(data["end_time"], "%H:%M:%S")

        elapsed = (t1 - t0).total_seconds()
        if elapsed < 0:
            elapsed += 24 * 3600

        data["elapsed_time"] = elapsed
        data["network_time"] = (
            elapsed - data["total_preprocess_time"]
        )

    except Exception:
        return None

    return data


############################################################
# Read all logs
############################################################

def parse_kdtree_log(log_path):
    """
    Parses kdtree.txt which is assumed to contain:
    - elapsed time
    - mIoU
    in a similar format to baseline/proposed logs.
    """

    data = {
        "elapsed_time": None,
        "miou": None,
    }

    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2})")
    miou_pattern = re.compile(r"so far miou is:\s*([0-9.]+)")

    last_miou = None
    start_time = None
    end_time = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            m = miou_pattern.search(line)
            if m:
                last_miou = float(m.group(1))

            # try to capture first timestamp as start
            if start_time is None:
                t = time_pattern.search(line)
                if t:
                    start_time = t.group(1)

            # keep updating end time
            t = time_pattern.search(line)
            if t:
                end_time = t.group(1)

    data["miou"] = last_miou

    # compute elapsed
    try:
        if start_time and end_time:
            t0 = datetime.strptime(start_time, "%H:%M:%S")
            t1 = datetime.strptime(end_time, "%H:%M:%S")

            elapsed = (t1 - t0).total_seconds()
            if elapsed < 0:
                elapsed += 24 * 3600

            data["elapsed_time"] = elapsed

    except Exception:
        pass

    return data
    
def collect_results(directory):

    directory = Path(directory)

    proposed = {}
    baseline = None
    kdtree = None   # NEW

    b_pattern = re.compile(r"B-(\d+)")

    for file in sorted(directory.glob("*.txt")):

        parsed = parse_log(file)

        if file.name.lower() == "baseline.txt":
            baseline = parsed
            continue

        # NEW: kdtree file
        if file.name.lower() == "kdtree.txt":
            kdtree = parse_log(file)
            continue

        if parsed is None:
            print(f"Skipping incomplete log: {file.name}")
            continue

        m = b_pattern.search(file.name)

        if m:
            B = int(m.group(1))
            proposed[B] = parsed

    return proposed, baseline, kdtree
############################################################
# Plot
############################################################

def plot_results(proposed, baseline, kdtree):

    if len(proposed) == 0:
        print("No completed proposed logs found.")
        return

    if baseline is None:
        print("Baseline not found or incomplete.")
        return

    Bs = sorted(proposed.keys())

    elapsed = np.array(
        [proposed[b]["elapsed_time"] for b in Bs],
        dtype=float,
    )

    preprocess = np.array(
        [proposed[b]["total_preprocess_time"] for b in Bs],
        dtype=float,
    )

    network = np.array(
        [proposed[b]["network_time"] for b in Bs],
        dtype=float,
    )

    miou = np.array(
        [proposed[b]["miou"] for b in Bs],
        dtype=float,
    )

    baseline_time = baseline["elapsed_time"]
    baseline_miou = baseline["miou"]

    # KDTree values
    kdtree_time = kdtree.get("elapsed_time") if kdtree else None
    kdtree_miou = kdtree.get("miou") if kdtree else None

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    ########################################################
    # Runtime plot
    ########################################################

    ax = axes[0]

    width = 0.7

    ax.bar(
        Bs,
        network,
        width,
        label="Network",
    )

    ax.bar(
        Bs,
        preprocess,
        width,
        bottom=network,
        label="Preprocessing",
    )

    # Baseline line
    ax.axhline(
        baseline_time,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Voxel-based",
    )

    # KDTree line
    if kdtree_time is not None:
        ax.axhline(
            kdtree_time,
            color="blue",
            linestyle="--",
            linewidth=2,
            label="KDTree",
        )

    ax.set_xlabel("B")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Runtime")
    ax.legend()

    ########################################################
    # mIoU plot
    ########################################################

    ax = axes[1]

    ax.plot(
        Bs,
        miou,
        marker="o",
        linewidth=2,
        label="KD-tree",
    )

    # Baseline line
    ax.axhline(
        baseline_miou,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Voxel-based",
    )

    # KDTree line
    if kdtree_miou is not None:
        ax.axhline(
            kdtree_miou,
            color="blue",
            linestyle="--",
            linewidth=2,
            label="KDTree",
        )

    ax.set_xlabel("B")
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Segmentation Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.savefig(
        "runtime_vs_B.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

############################################################
# Main
############################################################

directory = str(Path(__file__).resolve().parents[2] / "revsion" / "reviewer1" / "comment2")

proposed, baseline, kdtree = collect_results(directory)

print("\nCompleted experiments:")
for B in sorted(proposed):
    print(f"B={B}: {proposed[B]}")

if baseline is not None:
    print("\nBaseline:")
    print(baseline)
print("\nKdtree: ", kdtree)
plot_results(proposed, baseline, kdtree)

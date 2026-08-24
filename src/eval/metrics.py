"""Per-run metrics logging and loss-curve plotting.

Appends rows to ``<run_dir>/metrics.csv`` during training (no dependencies), and
renders ``<run_dir>/loss.png`` from that CSV (needs matplotlib). Keeping the CSV
separate means the data is always saved even if plotting isn't available.
"""

import csv
from pathlib import Path


def log_metrics(run_dir: Path, row: dict, filename: str = "metrics.csv") -> Path:
    """Append one row to the run's metrics CSV, writing the header once.

    Args:
        run_dir: The run directory.
        row: Column -> value for this record (keys define the header).
        filename: CSV filename within the run dir.

    Returns:
        Path to the CSV.
    """
    path = Path(run_dir) / filename
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def plot_losses(run_dir: Path, x: str = "step", filename: str = "metrics.csv",
                out: str = "loss.png") -> Path | None:
    """Plot train/val loss vs ``x`` from the metrics CSV; save as a PNG.

    Returns the PNG path, or None if matplotlib is unavailable or there's no data.
    """
    path = Path(run_dir) / filename
    if not path.exists():
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (metrics.csv still saved)")
        return None

    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ("train_loss", "val_loss"):
        pts = [(float(r[x]), float(r[col])) for r in rows if r.get(col) not in (None, "")]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", markersize=3, label=col)
    ax.set_xlabel(x)
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(run_dir) / out
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_series(run_dir, x="step", panels=None, filename="metrics.csv", out="curves.png"):
    """Plot stacked panels of metric columns vs ``x`` from the run's CSV.

    ``panels`` is a list of ``(ylabel, [column, ...])`` -- one stacked subplot per
    panel, the columns in a panel sharing a y-axis. For the RL/DPO curves that
    ``plot_losses`` can't render (reward/KL for GRPO, reward/value-loss for PPO,
    loss/margin for DPO). Returns the PNG path, or None if unplottable.
    """
    path = Path(run_dir) / filename
    if not path.exists() or not panels:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (metrics.csv still saved)")
        return None

    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 2.6 * len(panels)),
                             sharex=True, squeeze=False)
    for ax, (ylabel, cols) in zip(axes[:, 0], panels):
        drew = False
        for col in cols:
            pts = [(float(r[x]), float(r[col])) for r in rows if r.get(col) not in (None, "")]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", markersize=2, label=col)
                drew = True
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if drew and len(cols) > 1:
            ax.legend()
    axes[-1, 0].set_xlabel(x)
    out_path = Path(run_dir) / out
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
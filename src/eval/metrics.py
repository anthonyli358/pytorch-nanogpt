import csv
from pathlib import Path


def log_metrics(run_dir: Path, row: dict, filename: str = "metrics.csv") -> Path:
    """
    Append one row to the run's metrics CSV, writing the header once.

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


def plot_losses(
    run_dir: Path, x: str = "step", filename: str = "metrics.csv", out: str = "loss.png"
) -> Path | None:
    """
    Plot train/val loss from the metrics CSV and save as a PNG.

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
        pts = [
            (float(r[x]), float(r[col])) for r in rows if r.get(col) not in (None, "")
        ]
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


def _ema(ys, alpha):
    """Exponential moving average over a sequence (returns a list of the same length)."""
    out, m = [], ys[0]
    for y in ys:
        m = alpha * y + (1 - alpha) * m
        out.append(m)
    return out


def plot_series(
    run_dir,
    x="step",
    panels=None,
    filename="metrics.csv",
    out="curves.png",
    ema_cols=None,
    ema_alpha=0.3,
):
    """
    Plot stacked panels of metric columns from the run's CSV.
    Each panel is a single metric, 'panels' is a list of (ylabel, [column, ...])

    Columns named in `ema_cols` are drawn as a faint raw series plus a bold
    exponential moving average.

    Returns the PNG path, or None if unplottable.
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

    ema_cols = set(ema_cols or ())
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(7, 2.6 * len(panels)), sharex=True, squeeze=False
    )
    for ax, (ylabel, cols) in zip(axes[:, 0], panels):
        labeled = False
        for col in cols:
            pts = [
                (float(r[x]), float(r[col]))
                for r in rows
                if r.get(col) not in (None, "")
            ]
            if not pts:
                continue
            xs, ys = zip(*pts)
            if col in ema_cols and len(ys) > 1:
                # bold EMA first with the faint raw line behind it
                line, = ax.plot(xs, _ema(ys, ema_alpha), linewidth=2.0, label=f"{col} (EMA)")
                ax.plot(xs, ys, color=line.get_color(), alpha=0.25, linewidth=0.9)
                labeled = True
            else:
                ax.plot(xs, ys, marker="o", markersize=3, label=col)
                labeled = labeled or len(cols) > 1
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if labeled:
            ax.legend()
    axes[-1, 0].set_xlabel(x)
    out_path = Path(run_dir) / out
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path

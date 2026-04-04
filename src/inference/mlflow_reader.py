"""
Read MLflow experiment data directly from the mlruns/ flat-file backend.

No MLflow dependency — just YAML + text file parsing.  Returns deduplicated
experiment runs with metrics, params, tags, and artifact paths for the UI.
"""

import yaml
from pathlib import Path

from src.utils import create_logger

logger = create_logger("mlflow_reader")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MLRUNS_DIR = _PROJECT_ROOT / "mlruns"


def _read_file(path: Path) -> str:
    """Read a small text file, return empty string on failure."""
    try:
        return path.read_text().strip()
    except Exception:
        return ""


def _read_metric(path: Path) -> float | None:
    """Read an MLflow metric file (format: 'timestamp value step').

    Args:
        path: Path to a metric file.

    Returns:
        The metric value as a float, or None if unreadable.
    """
    try:
        text = path.read_text().strip()
        if text:
            return float(text.split()[1])
    except Exception:
        pass
    return None


def _find_artifact(run_dir: Path, subpath: str) -> str | None:
    """Find a single file under a run's artifact subdirectory.

    Args:
        run_dir: Path to the run directory.
        subpath: Relative path under artifacts/ (e.g. 'plots/confusion_matrix').

    Returns:
        Filename (not full path) if found, else None.
    """
    try:
        art_dir = run_dir / "artifacts" / subpath
        if art_dir.exists():
            files = [f for f in art_dir.iterdir() if f.is_file()]
            if files:
                return files[0].name
    except Exception:
        pass
    return None


def _parse_run(run_dir: Path, experiment_id: str) -> dict | None:
    """Parse a single MLflow run directory into a dict.

    Args:
        run_dir: Path to the run directory (contains meta.yaml, metrics/, params/, tags/).
        experiment_id: Parent experiment ID.

    Returns:
        Dict with run metadata, metrics, params, tags, artifact info.  None if invalid.
    """
    try:
        meta_path = run_dir / "meta.yaml"
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        run_name = meta.get("run_name", "unknown")
        run_id = meta.get("run_id", run_dir.name)

        # Read metrics
        metrics_dir = run_dir / "metrics"
        metrics = {}
        if metrics_dir.exists():
            for mf in metrics_dir.iterdir():
                val = _read_metric(mf)
                if val is not None:
                    metrics[mf.name] = round(val, 4)

        # Skip registry-only runs (no real metrics)
        if not metrics:
            return None

        # Read params
        params_dir = run_dir / "params"
        params = {}
        if params_dir.exists():
            for pf in params_dir.iterdir():
                params[pf.name] = _read_file(pf)

        # Read tags
        tags_dir = run_dir / "tags"
        tags = {}
        if tags_dir.exists():
            for tf in tags_dir.iterdir():
                tags[tf.name] = _read_file(tf)

        # Artifact paths (relative to run dir for serving)
        cm_file = _find_artifact(run_dir, "plots/confusion_matrix")
        roc_file = _find_artifact(run_dir, "plots/roc_curves")

        return {
            "run_id": run_id,
            "run_name": run_name,
            "experiment_id": experiment_id,
            "model_type": tags.get("model_type", ""),
            "variant": tags.get("variant", ""),
            "n_classes": tags.get("n_classes", ""),
            "n_features": tags.get("n_features", ""),
            "seq_len": tags.get("seq_len", ""),
            "metrics": metrics,
            "params": params,
            "artifacts": {
                "confusion_matrix": f"{experiment_id}/{run_id}/artifacts/plots/confusion_matrix/{cm_file}" if cm_file else None,
                "roc_curves": f"{experiment_id}/{run_id}/artifacts/plots/roc_curves/{roc_file}" if roc_file else None,
            },
        }

    except Exception as e:
        logger.error(f"Failed to parse run {run_dir}: {e}")
        return None


def load_experiments() -> dict:
    """Load all MLflow experiments, deduplicate by (run_name, metrics).

    Returns:
        Dict with experiment name and list of unique runs.
    """
    try:
        if not MLRUNS_DIR.exists():
            return {"error": "mlruns/ directory not found"}

        # Find experiment directories (skip 0/default and models/)
        experiments = []
        for exp_dir in MLRUNS_DIR.iterdir():
            if not exp_dir.is_dir() or exp_dir.name in ("0", "models", ".trash"):
                continue
            meta_path = exp_dir / "meta.yaml"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                exp_meta = yaml.safe_load(f)

            exp_name = exp_meta.get("name", exp_dir.name)
            exp_id = exp_meta.get("experiment_id", exp_dir.name)

            # Parse all runs
            runs = []
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name in ("models",):
                    continue
                run = _parse_run(run_dir, exp_id)
                if run:
                    runs.append(run)

            # Deduplicate: keep first run per (run_name, metric signature)
            seen = set()
            unique_runs = []
            for run in runs:
                key = (run["run_name"], tuple(sorted(run["metrics"].items())))
                if key not in seen:
                    seen.add(key)
                    unique_runs.append(run)

            # Sort by test_macro_f1 descending
            unique_runs.sort(
                key=lambda r: r["metrics"].get("test_macro_f1", 0),
                reverse=True,
            )

            experiments.append({
                "experiment_name": exp_name,
                "experiment_id": exp_id,
                "total_runs": len(runs),
                "unique_runs": len(unique_runs),
                "runs": unique_runs,
            })

        return {
            "experiments": experiments,
            "n_experiments": len(experiments),
        }

    except Exception as e:
        logger.error(f"Failed to load experiments: {e}")
        return {"error": f"Failed to load experiments: {e}"}

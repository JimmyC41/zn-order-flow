"""Reproduce the May–August ZN quote-filter, markout and order-count checks."""

from datetime import datetime
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
SUPPLEMENTS = ANALYSIS / "validation_supplements"
DATA = ROOT / "data"


def check_inputs():
    raw = DATA / "databento" / "ZN_tbbo"
    names = [f"2026{month:02}01_2026{month + 1:02}01.parquet" for month in range(5, 9)]
    missing = [name for name in names if not (raw / name).is_file()]
    if missing:
        raise SystemExit("Place the shared monthly files in data/databento/ZN_tbbo/. Missing: "
                         + ", ".join(missing))
    packages = ("numpy", "pandas", "scipy", "pyarrow", "numba", "duckdb")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing Python packages: " + ", ".join(missing)
                         + ". See analysis/validation_supplements/README.md.")


def run_step(script, *args):
    result = subprocess.run([sys.executable, str(script), *map(str, args)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    if result.returncode:
        print(result.stdout, file=sys.stderr)
        raise SystemExit(f"{script.name} failed. No new results were published.")


def main():
    check_inputs()
    from validation_supplements.export_validation_diagnostic import new_output_directory

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = DATA / "supplementary_validation_runs" / stamp
    with TemporaryDirectory(prefix="zn-validation-") as directory:
        work = Path(directory)
        export = work / "export"
        quotes = work / "quotes"
        forecast = work / "forecast"
        print("1/3  Reconstructing orders from the four validation files", flush=True)
        run_step(SUPPLEMENTS / "export_validation_diagnostic.py", "--out", export)

        print("2/3  Comparing quote filters and checking markout timing", flush=True)
        events = export / "validation_events.csv.gz"
        model = export / "frozen_model.json"
        source = export / "export_manifest.json"
        run_step(SUPPLEMENTS / "audit.py", "--input", events, "--model", model,
                 "--data-kind", "validation", "--output-dir", quotes, "--bootstrap", 2000, "--seed", 3666)
        run_step(SUPPLEMENTS / "supplemental_report.py", "--results", quotes, "--official",
                 ROOT / "output/tables/validation_quoting_rule.csv", "--input", events,
                 "--model", model, "--export-manifest", source)

        print("3/3  Scoring order-count forecasts", flush=True)
        run_step(SUPPLEMENTS / "run_validation_forecast.py", "--orders", export / "validation_orders.csv.gz",
                 "--model", model, "--config", export / "frozen_config.json", "--news", export / "news_releases.csv",
                 "--output", forecast, "--input-label", "validation_export")
        run_step(SUPPLEMENTS / "summarize_results.py", forecast, "--export-manifest", source)

        with new_output_directory(destination) as out:
            shutil.copyfile(quotes / "interpretation/news_only_increment_table.csv", out / "news_only_increment_table.csv")
            shutil.copyfile(quotes / "sensitivity.csv", out / "sensitivity.csv")
            shutil.copyfile(forecast / "comparison.tsv", out / "count_forecast.tsv")
            for detail in (quotes / "lag_diagnostics.csv",
                           quotes / "interpretation/official_reconciliation.csv"):
                shutil.copyfile(detail, out / detail.name)
    print(f"Results: {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

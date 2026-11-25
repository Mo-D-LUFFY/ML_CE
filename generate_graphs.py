#!/usr/bin/env python3
"""CS725 counterfactual experiment: publication-friendly charts from saved outputs.

No model training and no fabricated graph values. Reads files written by run_demo.py.

From inside cs725_starter/:
    python generate_graphs.py
    python generate_graphs.py --results-dir outputs --output-dir graphs --format both

Creates PNG and SVG figures, a browser-friendly HTML gallery, a derived CSV,
and a JSON manifest. Run it again after any new run_demo.py experiment.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Compatible with WSL/SSH/server environments.
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Consistent, restrained report palette (works well on light slides).
COLORS = {
    "logistic_regression": "#365E86",
    "random_forest": "#148B84",
    "unconstrained": "#365E86",
    "constrained": "#148B84",
    "flip": "#527BA2",
    "feasible": "#148B84",
    "blocked": "#C55F53",
    "caution": "#DFAB58",
    "pass": "#148B84",
    "no_candidate": "#9CA6B1",
    "other": "#B29ABA",
}
MODELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
METHODS = {"unconstrained": "Unconstrained", "constrained": "Constrained"}


def nice_model(value: str) -> str:
    return MODELS.get(str(value), str(value).replace("_", " ").title())


def nice_method(value: str) -> str:
    return METHODS.get(str(value), str(value).replace("_", " ").title())


def read_json(path: Path, required: bool = False):
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing {path}. Run `python run_demo.py --dataset synthetic` first.")
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_files(folder: Path):
    for f in ["classifier_metrics.json", "comparison_summary.csv", "per_query_results.csv"]:
        if not (folder / f).is_file():
            raise FileNotFoundError(f"Missing {folder / f}. Run `python run_demo.py --dataset synthetic` first.")
    metrics = read_json(folder / "classifier_metrics.json", required=True)
    summary = pd.read_csv(folder / "comparison_summary.csv")
    records = pd.read_csv(folder / "per_query_results.csv", dtype={"query_id": "string"})
    meta = read_json(folder / "run_metadata.json") or {}
    cases = read_json(folder / "query_examples.json")
    example = read_json(folder / "demo_example.json")

    expected_summary = {"classifier", "generator", "queries", "flip_success_count", "feasible_flip_count"}
    expected_rows = {"classifier", "generator", "query_id", "flip_success", "feasible_flip", "blocked", "caution", "cost_units"}
    if not expected_summary.issubset(summary.columns):
        raise ValueError(f"comparison_summary.csv missing columns: {sorted(expected_summary - set(summary.columns))}")
    if not expected_rows.issubset(records.columns):
        raise ValueError(f"per_query_results.csv missing columns: {sorted(expected_rows - set(records.columns))}")

    # Re-derive rates from counts for correctness even if a summary's rounded rates change.
    if (summary["queries"] <= 0).any():
        raise ValueError("Each comparison_summary.csv row must have >0 queries")
    summary["flip_rate"] = summary["flip_success_count"] / summary["queries"]
    summary["feasible_rate"] = summary["feasible_flip_count"] / summary["queries"]
    summary["conditional_feasibility"] = np.where(
        summary["flip_success_count"] > 0,
        summary["feasible_flip_count"] / summary["flip_success_count"].replace(0, np.nan),
        np.nan,
    )
    return metrics, summary, records, meta, cases, example


class ChartBook:
    def __init__(self, folder: Path, image_format: str, dpi: int, metadata: dict):
        self.folder = folder
        self.image_format = image_format
        self.dpi = dpi
        self.files = []
        self.meta = metadata
        self.tag = ("SYNTHETIC DATA — software demonstration, not UCI Adult results"
                    if metadata.get("is_synthetic") is True else
                    "UCI ADULT DATA" if metadata.get("dataset") == "adult" else
                    "Dataset not identified — inspect run_metadata.json")
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
            "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "axes.spines.top": False, "axes.spines.right": False,
            "savefig.facecolor": "white", "axes.axisbelow": True,
        })

    def canvas(self, title: str, description: str, *, width=11.7, height=6.9):
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes((0.10, 0.20, 0.84, 0.61))
        fig.text(0.10, 0.945, title, ha="left", va="top", size=16, weight="bold", color="#182C3D")
        fig.text(0.10, 0.889, description, ha="left", va="top", size=10, color="#526578")
        ax.grid(axis="y", color="#E9EDF0", linewidth=0.8)
        return fig, ax

    def save(self, fig, basename: str, caption: str, note: str = ""):
        disclaimer = self.tag
        if note:
            disclaimer = f"{disclaimer}   •   {note}"
        fig.text(0.10, 0.060, disclaimer, va="center", ha="left", size=8.2,
                 color="#73808D", wrap=True)
        formats = ["png", "svg"] if self.image_format == "both" else [self.image_format]
        for ext in formats:
            fig.savefig(self.folder / f"{basename}.{ext}", dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
        self.files.append({"chart": basename, "caption": caption, "formats": formats})
        print(f"  generated {basename} ({', '.join(formats)})")


def add_bar_labels(ax, bars, *, proportion=False, count=None):
    for j, bar in enumerate(bars):
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        text = f"{h:.0%}" if proportion else f"{h:.2f}"
        if count is not None:
            text += f"\n(n={int(count[j])})"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", size=8.7)


def plot_quality(book: ChartBook, metrics: dict):
    names = list(metrics)
    fields = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced\naccuracy"),
              ("positive_f1", "Positive-class\nF1"), ("roc_auc", "ROC AUC")]
    fig, ax = book.canvas("01  •  Classifier performance", "Held-out test-set results; 1.00 = perfect score")
    x = np.arange(len(fields))
    w = min(0.34, 0.75 / max(len(names), 1))
    for i, model in enumerate(names):
        vals = [float(metrics[model].get(key, np.nan)) for key, _ in fields]
        offset = (i - (len(names) - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w * 0.91, label=nice_model(model),
                      color=COLORS.get(model, None))
        add_bar_labels(ax, bars, proportion=True)
    ax.set_xticks(x, [label for _, label in fields])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Held-out score")
    ax.legend(loc="upper center", ncol=min(3, len(names)), frameon=False)
    book.save(fig, "01_model_metrics", "Held-out predictive performance",
              "Scores describe the fitted classifier, not the quality of recourse advice")


def plot_confusion(book: ChartBook, metrics: dict):
    for i, (model, entry) in enumerate(metrics.items(), 2):
        conf = entry.get("confusion_matrix")
        if not conf:
            continue
        required = ("TN", "FP", "FN", "TP")
        if any(key not in conf for key in required):
            continue
        matrix = np.array([[conf["TN"], conf["FP"]], [conf["FN"], conf["TP"]]], dtype=int)
        fig, ax = book.canvas(f"{i:02d}  •  Confusion matrix — {nice_model(model)}",
                              "Held-out counts: rows are actual categories; columns are model predictions")
        ax.grid(False)
        image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0,
                          vmax=max(1, int(matrix.max())))
        ax.set_xticks([0, 1], ["≤$50K", ">$50K"])
        ax.set_yticks([0, 1], ["≤$50K", ">$50K"])
        ax.set_xlabel("Predicted income class")
        ax.set_ylabel("Actual income class")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        for (row, col), value in np.ndenumerate(matrix):
            tag = [["TN", "FP"], ["FN", "TP"]][row][col]
            ax.text(col, row, f"{value:,}\n{tag}", ha="center", va="center",
                    weight="bold", fontsize=15,
                    color="white" if value > 0.55 * matrix.max() else "#1B344C")
        fig.colorbar(image, ax=ax, label="Number of test records", shrink=0.81, pad=0.02)
        book.save(fig, f"{i:02d}_confusion_{model}", f"Confusion matrix for {nice_model(model)}",
                  f"n={int(matrix.sum())} held-out test records")


def group_labels(summary):
    return [f"{nice_model(row.classifier)}\n{nice_method(row.generator)}"
            for row in summary.itertuples(index=False)]


def plot_success_rates(book, summary):
    fig, ax = book.canvas("04  •  Prediction flips versus feasible flips",
                          "Both bars use all queried profiles as their denominator")
    x = np.arange(len(summary))
    w = 0.36
    for offset, col, color, label, numerator in [
        (-w / 2, "flip_rate", COLORS["flip"], "Any prediction flip", "flip_success_count"),
        (w / 2, "feasible_rate", COLORS["feasible"], "Flip + hard-policy feasibility", "feasible_flip_count"),
    ]:
        bars = ax.bar(x + offset, summary[col], width=w, label=label, color=color)
        for bar, c, n in zip(bars, summary[numerator], summary["queries"]):
            height = float(bar.get_height())
            label_text = f"{height:.0%}\n{c}/{n}"
            if height > 0.12:
                ax.annotate(label_text, (bar.get_x() + bar.get_width()/2, height - 0.02),
                            ha="center", va="top", fontsize=8.5, color="white", weight="bold")
            else:
                ax.annotate(label_text, (bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, group_labels(summary), fontsize=9)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Fraction of queried negative profiles")
    ax.legend(ncol=2, frameon=False, loc="upper center")
    book.save(fig, "04_flip_vs_feasible", "Predicted class flip versus hard-constraint passing",
              "Generator search spaces differ; this is a descriptive comparison, not a causal constraint effect")


def status_of(row):
    if not bool(row["flip_success"]):
        return "No returned flip"
    if bool(row["blocked"]):
        return "Flip + blocked"
    if bool(row["feasible_flip"]):
        return "Flip + caution" if bool(row["caution"]) else "Flip + pass"
    return "Other outcome"


def plot_outcomes(book, rows):
    counts = rows.copy()
    counts["status"] = counts.apply(status_of, axis=1)
    groups = list(counts.groupby(["classifier", "generator"], sort=False))
    if not groups:
        return pd.DataFrame()
    labels = [f"{nice_model(a)}\n{nice_method(b)}" for (a, b), _ in groups]
    categories = ["No returned flip", "Flip + blocked", "Flip + caution", "Flip + pass", "Other outcome"]
    palette = [COLORS["no_candidate"], COLORS["blocked"], COLORS["caution"], COLORS["pass"], COLORS["other"]]
    fig, ax = book.canvas("05  •  Counterfactual outcome breakdown",
                          "A mutually exclusive verdict for every attempted query (100% stacked bars)")
    ax.set_position((0.15, 0.20, 0.79, 0.54))  # Reserve space above for the legend.
    left = np.zeros(len(groups))
    aggregate = []
    for cat, color in zip(categories, palette):
        values = []
        for (model, generator), group in groups:
            n = int((group["status"] == cat).sum())
            values.append(100 * n / len(group))
            aggregate.append({"classifier": model, "generator": generator,
                              "status": cat, "count": n, "queries": int(len(group))})
        vals = np.asarray(values)
        bars = ax.barh(np.arange(len(groups)), vals, left=left, color=color, label=cat, height=0.57)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v >= 10:
                ax.text(left[j] + v/2, j, f"{v:.0f}%", ha="center", va="center",
                        size=9, weight="bold", color="#172536" if cat != "Flip + blocked" else "white")
        left += vals
    ax.set_yticks(range(len(groups)), labels)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of queried profiles")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E9EDF0")
    ax.grid(axis="y", visible=False)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, ncol=3, loc="upper left", bbox_to_anchor=(0.10, 0.845),
               frameon=False, fontsize=8.5)
    book.save(fig, "05_query_verdicts", "Status of every counterfactual query",
              "PASS only means the specified rules raised no warning; it does not prove real-world viability")
    return pd.DataFrame(aggregate)


def plot_costs(book, records):
    eligible = records[(records["feasible_flip"] == 1) & records["cost_units"].notna()].copy()
    if eligible.empty:
        print("  skipped cost distribution: no feasible candidates with recorded cost")
        return
    grouped = list(eligible.groupby(["classifier", "generator"], sort=False))
    samples = [entry["cost_units"].astype(float).to_numpy() for _, entry in grouped]
    labels = [f"{nice_model(m)}\n{nice_method(g)}\n(n={len(s)})"
              for ((m, g), _), s in zip(grouped, samples)]
    fig, ax = book.canvas("06  •  Estimated effort of feasible candidates",
                          "Only successfully flipped candidates passing all hard constraints are included")
    bp = ax.boxplot(samples, tick_labels=labels, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#132536"},
                    medianprops={"linewidth": 2, "color": "#132536"},
                    flierprops={"markersize": 4})
    for patch, ((_, gen), _) in zip(bp["boxes"], grouped):
        patch.set_facecolor(COLORS.get(gen, "#8098B0"))
        patch.set_alpha(0.70)
    ax.set_ylabel("Estimated effort (arbitrary policy units)")
    ax.set_ylim(bottom=0)
    book.save(fig, "06_feasible_effort", "Distribution of estimated effort for feasible flips",
              "Missing blocked/no-candidate costs are excluded; units are NOT currency")


def plot_warnings(book, cases):
    if not isinstance(cases, dict):
        print("  skipped warning reasons: query_examples.json missing")
        return
    stats = Counter()
    method_names = []
    for model, examples in cases.items():
        if not isinstance(examples, list):
            continue
        for query in examples:
            for gen in ["unconstrained", "constrained"]:
                obj = query.get(gen)
                if not isinstance(obj, dict):
                    continue
                if gen not in method_names:
                    method_names.append(gen)
                for warning in obj.get("warnings", []):
                    code = warning.get("code", "UNKNOWN")
                    stats[(gen, code)] += 1
    codes = sorted({code for _, code in stats})
    if not codes:
        print("  skipped warning reasons: no warnings in query_examples.json")
        return
    names = [code.replace("_", " ").title() for code in codes]
    fig, ax = book.canvas("07  •  Why the validator raised warnings",
                          "Counts of warning codes across both classifiers; one candidate can trigger multiple codes")
    x = np.arange(len(codes))
    w = 0.80 / len(method_names)
    for i, method in enumerate(method_names):
        vals = [stats[(method, code)] for code in codes]
        b = ax.bar(x + (i - (len(method_names)-1)/2) * w, vals, width=w*0.92,
                   color=COLORS.get(method), label=nice_method(method))
        for bar in b:
            if bar.get_height():
                ax.annotate(f"{int(bar.get_height())}",
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, names, rotation=10, ha="center")
    ax.set_ylabel("Number of flagged candidate warnings")
    ax.legend(frameon=False)
    ymax = max(stats.values())
    ax.set_ylim(0, max(3, ymax * 1.20))
    book.save(fig, "07_warning_codes", "Frequency of validator warning codes",
              "Counts are across candidates; warning rules are explicit assumptions, not learned truth")


def plot_example(book, example):
    if not isinstance(example, dict):
        print("  skipped example: demo_example.json missing")
        return
    records = [example.get("unconstrained_counterfactual"), example.get("constrained_counterfactual")]
    first = next((r for r in records if isinstance(r, dict)), None)
    if first is None:
        return
    vals = [float(first["original_probability_gt_50k"])]
    names = ["Original"]
    colors = [COLORS["no_candidate"]]
    for method, item in [("unconstrained", records[0]), ("constrained", records[1])]:
        if isinstance(item, dict):
            vals.append(float(item["counterfactual_probability_gt_50k"]))
            names.append(f"{nice_method(method)}\n{item.get('verdict', 'UNKNOWN')}")
            colors.append(COLORS.get(item.get("verdict", "").lower(), COLORS.get(method)))
    fig, ax = book.canvas("08  •  One counterfactual case, explained",
                          f"Case {example.get('query_id', '?')} — {nice_model(example.get('classifier', 'model'))}; score for >$50K")
    bars = ax.bar(np.arange(len(vals)), vals, width=0.59, color=colors)
    ax.axhline(0.5, color="#C55F53", linewidth=1.5, linestyle="--", label="0.50 decision threshold")
    for bar in bars:
        ax.annotate(f"{bar.get_height() * 100:.1f}%",
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)), names)
    ax.set_ylim(0, max(0.83, max(vals) + 0.15))
    ax.set_ylabel("Classifier score for higher-income category")
    ax.legend(frameon=False)
    book.save(fig, "08_before_after_example", "Before-and-after scores for one real generated test case",
              "Model scores are not verified probabilities that the person's actual income will increase")


def write_gallery(folder, book, summary, statuses):
    dataset_description = html.escape(book.tag)
    cards = []
    for item in book.files:
        filename = item["chart"] + ".png"
        preview = (f'<img loading="lazy" src="{html.escape(filename)}" '
                   f'alt="{html.escape(item["caption"])}">' if "png" in item["formats"] else
                   '<p>Image available as SVG</p>')
        link_ext = "png" if "png" in item["formats"] else "svg"
        cards.append(f'<article><a href="{item["chart"]}.{link_ext}">{preview}</a>'
                     f'<h2>{html.escape(item["caption"])}</h2>'
                     f'<p><a href="{item["chart"]}.{link_ext}">View figure</a></p></article>')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS725 — Counterfactual Results Gallery</title>
<style>body{{margin:0;background:#f5f7f9;color:#20364a;font-family:system-ui,Arial,sans-serif}}
main{{max-width:1250px;margin:auto;padding:32px}}h1{{font-size:29px;margin:0 0 10px}}
.subtitle{{color:#5c6d7e;line-height:1.5}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:19px;margin-top:24px}}
article{{background:#fff;border:1px solid #dde4e9;border-radius:12px;overflow:hidden}}
article img{{width:100%;height:auto;display:block}}article h2{{font-size:16px;padding:0 18px}}
article p{{padding:0 18px 12px}}a{{color:#195f8b}}footer{{color:#65778a;margin:25px 0}}
@media(max-width:540px){{.gallery{{grid-template-columns:1fr}} main{{padding:14px}}}}</style></head>
<body><main><h1>CS725 — Counterfactual experiment results</h1>
<p class="subtitle"><b>{dataset_description}</b><br>Charts were generated from saved CSV/JSON outputs, not hand-entered numbers.
<b>Interpretation:</b> flips do not establish real-world effects; constrained and unconstrained starter searches have different candidate sets.</p>
<section class="gallery">{''.join(cards)}</section>
<footer>Run <code>python generate_graphs.py</code> again after re-running the models to refresh every plot.</footer></main></body></html>'''
    (folder / "index.html").write_text(page, encoding="utf-8")
    summary.to_csv(folder / "chart_summary.csv", index=False)
    if not statuses.empty:
        statuses.to_csv(folder / "outcome_counts.csv", index=False)
    manifest = {"source_dataset": book.meta.get("dataset", "unknown"),
                "is_synthetic": book.meta.get("is_synthetic"), "figures": book.files,
                "limitations": [
                    "Counterfactual prediction flips do not establish causal effects.",
                    "Starter unconstrained and constrained generators have unequal candidate/search budgets.",
                    "Effort is a made-up policy score, not financial cost.",
                    "Failed search does not prove no globally feasible counterfactual exists.",
                ]}
    (folder / "charts_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Turn saved CS725 model/recourse outputs into graphs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs",
                        help="Directory containing run_demo.py results (default: ./outputs)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "graphs",
                        help="Where to save graph files (default: ./graphs)")
    parser.add_argument("--format", choices=["png", "svg", "both"], default="both")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    results_folder = args.results_dir.expanduser().resolve()
    output_folder = args.output_dir.expanduser().resolve()
    try:
        metrics, summary, rows, meta, cases, example = load_files(results_folder)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    output_folder.mkdir(parents=True, exist_ok=True)
    book = ChartBook(output_folder, args.format, args.dpi, meta)
    print(f"Reading: {results_folder}\nDataset: {book.tag}\nWriting: {output_folder}")
    plot_quality(book, metrics)
    plot_confusion(book, metrics)
    plot_success_rates(book, summary)
    statuses = plot_outcomes(book, rows)
    plot_costs(book, rows)
    plot_warnings(book, cases)
    plot_example(book, example)
    write_gallery(output_folder, book, summary, statuses)
    print(f"\nDone. {len(book.files)} figures, saved as {args.format}. Open {output_folder / 'index.html'}")


if __name__ == "__main__":
    main()

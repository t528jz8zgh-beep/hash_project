#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate paper-ready figures + summary tables + report snippets
from benchmark CSVs produced by bench_csv.py.

Usage:
  cd ~/Desktop/hash_project
  python3 make_report_artifacts.py \
    --alpha_csv alpha_sweep.csv \
    --nsweep_csv n_sweep_pow2.csv \
    --out_dir analysis_out

Optional:
  --variance_csv variance_check.csv
  --alpha_extreme_csv alpha_extreme.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def read_bench_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "algo" in df.columns:
        df = df[df["algo"].astype(str).str.lower() != "algo"].copy()

    # make sure alpha exists
    if "alpha" not in df.columns and "delta" in df.columns:
        df["alpha"] = 1.0 - pd.to_numeric(df["delta"], errors="coerce")

    # numeric coercion for all non-string columns
    for c in df.columns:
        if c in ("algo", "notes", "search_mode"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["algo", "n", "delta", "alpha"], dropna=False)
    out = g.agg(
        trials=("trial", "nunique"),
        inserted_mean=("inserted", "mean"),
        inserted_std=("inserted", "std"),
        max_elements=("max_elements", "mean"),
        insert_fail_rate=("insert_fail", "mean"),
        insert_ns_per_op_mean=("insert_ns_per_op", "mean"),
        insert_ns_per_op_std=("insert_ns_per_op", "std"),
        insert_probes_p99_mean=("insert_probes_p99", "mean"),
        insert_probes_p99_std=("insert_probes_p99", "std"),
        hit_ns_per_op_mean=("hit_ns_per_op", "mean"),
        hit_ns_per_op_std=("hit_ns_per_op", "std"),
        hit_probes_p99_mean=("hit_probes_p99", "mean"),
        hit_probes_p99_std=("hit_probes_p99", "std"),
    ).reset_index()

    out["inserted_frac_mean"] = out["inserted_mean"] / out["max_elements"]
    return out


def plot_with_err(df: pd.DataFrame, x: str, y: str, yerr: str | None,
                  title: str, xlabel: str, ylabel: str,
                  outpath: Path, logx_base2: bool = False):
    plt.figure()
    for algo, sub in df.groupby("algo"):
        sub = sub.sort_values(x)
        xs = sub[x].values
        ys = sub[y].values
        if yerr and yerr in sub.columns:
            es = sub[yerr].values
            # If std is NaN (e.g., trials=1), matplotlib will ignore
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=str(algo))
        else:
            plt.plot(xs, ys, marker="o", label=str(algo))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logx_base2:
        plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def pick_key_points_alpha(alpha_sum: pd.DataFrame):
    # Pick the highest alpha row per algo (for short narrative)
    mx = alpha_sum["alpha"].max()
    sub = alpha_sum[alpha_sum["alpha"] == mx].copy()
    return mx, sub


def pick_key_points_n(ns_sum: pd.DataFrame):
    mx = ns_sum["n"].max()
    sub = ns_sum[ns_sum["n"] == mx].copy()
    return mx, sub


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_csv", required=True)
    ap.add_argument("--nsweep_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variance_csv", default=None)
    ap.add_argument("--alpha_extreme_csv", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # Load
    alpha_df = read_bench_csv(args.alpha_csv)
    ns_df = read_bench_csv(args.nsweep_csv)

    # Summarize
    alpha_sum = summarize(alpha_df)
    ns_sum = summarize(ns_df)

    alpha_sum.to_csv(tab_dir / "alpha_sweep_summary.csv", index=False)
    ns_sum.to_csv(tab_dir / "n_sweep_summary.csv", index=False)

    # Core figures (3–5)
    plot_with_err(
        alpha_sum, "alpha", "hit_probes_p99_mean", "hit_probes_p99_std",
        "Alpha sweep: p99 successful-query probe complexity (hit_probes_p99)",
        "alpha (=1-delta)", "p99 hit probes",
        fig_dir / "fig1_alpha_hit_probes_p99.png"
    )
    plot_with_err(
        alpha_sum, "alpha", "insert_probes_p99_mean", "insert_probes_p99_std",
        "Alpha sweep: p99 insertion probes (insert_probes_p99)",
        "alpha (=1-delta)", "p99 insert probes",
        fig_dir / "fig2_alpha_insert_probes_p99.png"
    )
    plot_with_err(
        alpha_sum, "alpha", "insert_ns_per_op_mean", "insert_ns_per_op_std",
        "Alpha sweep: insertion time (ns/op)",
        "alpha (=1-delta)", "insert ns/op",
        fig_dir / "fig3_alpha_insert_ns_per_op.png"
    )

    # Diagnostic: achieved load (important for Elastic validity)
    plot_with_err(
        alpha_sum, "alpha", "inserted_frac_mean", None,
        "Alpha sweep: achieved load (inserted / max_elements)",
        "alpha (=1-delta)", "inserted fraction",
        fig_dir / "diag_alpha_inserted_frac.png"
    )

    plot_with_err(
        ns_sum, "n", "hit_probes_p99_mean", "hit_probes_p99_std",
        "n sweep: p99 successful-query probe complexity (delta fixed)",
        "n", "p99 hit probes",
        fig_dir / "fig4_n_hit_probes_p99.png",
        logx_base2=True
    )
    plot_with_err(
        ns_sum, "n", "insert_ns_per_op_mean", "insert_ns_per_op_std",
        "n sweep: insertion time (ns/op, delta fixed)",
        "n", "insert ns/op",
        fig_dir / "fig5_n_insert_ns_per_op.png",
        logx_base2=True
    )

    plot_with_err(
        ns_sum, "n", "inserted_frac_mean", None,
        "n sweep: achieved load (inserted / max_elements)",
        "n", "inserted fraction",
        fig_dir / "diag_n_inserted_frac.png",
        logx_base2=True
    )

    # Optional: variance check plot
    if args.variance_csv:
        var_df = read_bench_csv(args.variance_csv)
        # Use raw per-trial points; plot errorbar by algo
        g = var_df.groupby("algo").agg(
            hit_p99_mean=("hit_probes_p99", "mean"),
            hit_p99_std=("hit_probes_p99", "std"),
            ins_ns_mean=("insert_ns_per_op", "mean"),
            ins_ns_std=("insert_ns_per_op", "std"),
        ).reset_index()

        plt.figure()
        plt.errorbar(g["algo"], g["hit_p99_mean"], yerr=g["hit_p99_std"], marker="o", capsize=4)
        plt.title("Variance check: hit_probes_p99 (mean ± std across trials)")
        plt.xlabel("algo")
        plt.ylabel("hit_probes_p99")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / "extra_variance_hit_probes_p99.png", dpi=200)
        plt.close()

    # Optional: extreme alpha / very small delta
    if args.alpha_extreme_csv:
        ex_df = read_bench_csv(args.alpha_extreme_csv)
        ex_sum = summarize(ex_df)
        # zoomed plot for alpha >= 0.90
        ex_sum_zoom = ex_sum[ex_sum["alpha"] >= 0.90].copy()
        plot_with_err(
            ex_sum_zoom, "alpha", "hit_probes_p99_mean", "hit_probes_p99_std",
            "Extreme-load alpha sweep (zoom): p99 hit probes, alpha>=0.90",
            "alpha (=1-delta)", "p99 hit probes",
            fig_dir / "extra_alpha_extreme_zoom_hit_p99.png"
        )

    # Report snippets (auto-filled with key points)
    max_alpha, alpha_key = pick_key_points_alpha(alpha_sum)
    max_n, n_key = pick_key_points_n(ns_sum)

    # Detect Elastic validity at key alpha
    elastic_row = alpha_key[alpha_key["algo"].str.lower() == "elastic"]
    elastic_ok = (len(elastic_row) == 0) or (float(elastic_row["inserted_frac_mean"].iloc[0]) >= 0.95 and float(elastic_row["insert_fail_rate"].iloc[0]) < 0.01)

    md = []
    md.append("# Report snippets (auto-generated)\n")
    md.append("## What is being compared\n")
    md.append(
        "- We report both **insertion cost** (insert probes/time) and **successful-query probe complexity** (hit probes).\n"
        "- For Elastic hashing, successful-query probe complexity is reported as the paper-defined index **φ(i,j)**; "
        "for Uniform/Funnel, hit probes correspond to the number of probes performed in the search procedure.\n"
    )

    md.append("## Results: alpha sweep (load sensitivity)\n")
    md.append(f"- Highest load in this dataset: **alpha={max_alpha:.2f}**.\n")
    md.append("- Figure references:\n")
    md.append("  - Fig.1: `fig1_alpha_hit_probes_p99.png`\n")
    md.append("  - Fig.2: `fig2_alpha_insert_probes_p99.png`\n")
    md.append("  - Fig.3: `fig3_alpha_insert_ns_per_op.png`\n")
    md.append("  - Diagnostic: `diag_alpha_inserted_frac.png` (must be close to 1.0 for valid same-load comparison)\n")

    if not elastic_ok:
        md.append("\n**Important:** Elastic did not reach the target load in at least the highest-alpha setting "
                  "(see `diag_alpha_inserted_frac.png` and `insert_fail_rate`). "
                  "In that case, Elastic-vs-(Uniform/Funnel) comparisons at the same (n, delta) are not valid; rerun after fixing Elastic.\n")

    md.append("\n## Results: n sweep (scalability)\n")
    md.append(f"- Largest n in this dataset: **n={int(max_n)}** (delta fixed).\n")
    md.append("- Figure references:\n")
    md.append("  - Fig.4: `fig4_n_hit_probes_p99.png`\n")
    md.append("  - Fig.5: `fig5_n_insert_ns_per_op.png`\n")
    md.append("  - Diagnostic: `diag_n_inserted_frac.png`\n")

    md.append("\n## Discussion pointers (template)\n")
    md.append("- Discuss **hit_probes_p99** as the main metric aligned with the paper’s probe-complexity definition for present keys.\n")
    md.append("- Discuss **insert_probes_p99** and **insert_ns/op** as the engineering cost trade-off.\n")
    md.append("- Use diagnostics to confirm each algorithm actually achieved the intended load.\n")

    (out_dir / "report_snippets.md").write_text("".join(md), encoding="utf-8")

    print(f"[OK] Wrote figures to: {fig_dir}")
    print(f"[OK] Wrote tables  to: {tab_dir}")
    print(f"[OK] Wrote report  to: {out_dir/'report_snippets.md'}")


if __name__ == "__main__":
    main()

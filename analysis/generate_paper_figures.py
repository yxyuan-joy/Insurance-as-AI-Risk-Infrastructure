#!/usr/bin/env python3
"""Regenerate manuscript figures from locally generated paired runs."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

try:
    import scienceplots

    HAS_SCIENCEPLOTS = scienceplots is not None
except Exception:
    HAS_SCIENCEPLOTS = False


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "runs" / "formal"
RULE_ABM_RUNS_DIR = REPO_ROOT / "runs" / "rule_abm"
RULE_ABM_RUN_NAMES = {
    42: "seed42",
    77: "seed77",
    202: "seed202",
}

BLUE = "#384ADD"
GREY = "#929EAB"
TEAL = "#9CBED1"
SAND = "#B8A9B8"
RED = "#DC6C6E"

DAYS_PER_ROUND = 3.0
CALENDAR_HORIZON_DAYS = 300
DAY_TICKS = [0, 60, 120, 180, 240, 300]

VENDOR_ORDER = ["Vendor_Alpha", "Vendor_Beta", "Vendor_Gamma", "Vendor_Delta"]
VENDOR_LABELS = {
    "Vendor_Alpha": "Vendor Alpha",
    "Vendor_Beta": "Vendor Beta",
    "Vendor_Gamma": "Vendor Gamma",
    "Vendor_Delta": "Vendor Delta",
}
VENDOR_RISK = {
    "Vendor_Alpha": 0.62,
    "Vendor_Beta": 0.82,
    "Vendor_Gamma": 1.10,
    "Vendor_Delta": 1.05,
}

INDUSTRY_LABELS = {
    "health_care": "HECA",
    "cons_discretionary": "CODI",
    "cons_staples": "COSE",
    "industrials": "INDU",
    "information_technology": "INTE",
    "communication_services": "COST",
    "energy": "ENER",
    "utilities": "UTIL",
    "financials": "FINA",
    "real_estate": "REES",
    "materials": "MATE",
}


def configure_style() -> None:
    if HAS_SCIENCEPLOTS:
        plt.style.use(["science", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_seed(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {path.name}")
    return int(match.group(1))


def read_run_table(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        path = run_dir / f"{filename}.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing required result table: {run_dir / filename}")
    return pd.read_csv(path)


def load_runs() -> dict[str, dict[int, dict[str, pd.DataFrame]]]:
    runs: dict[str, dict[int, dict[str, pd.DataFrame]]] = {"on": {}, "off": {}}
    for path in sorted(RESULTS_DIR.glob("seed*_*")):
        arm = path.name.rsplit("_", 1)[-1]
        if arm not in runs:
            continue
        seed = parse_seed(path)
        runs[arm][seed] = {
            "macro": read_run_table(path, "macro_daily.csv"),
            "firm": read_run_table(path, "firm_daily.csv"),
            "insurer": read_run_table(path, "insurer_daily.csv"),
            "decisions": read_run_table(path, "decisions.csv"),
            "events": pd.read_json(path / "events.jsonl.gz", lines=True, compression="gzip"),
        }
    if sorted(runs["on"]) != sorted(runs["off"]):
        raise RuntimeError(f"Paired seeds do not match: {sorted(runs['on'])} vs {sorted(runs['off'])}")
    return runs


def save(fig: plt.Figure, name: str, *, dpi: int = 300, tight: bool = True) -> None:
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=dpi, **kwargs)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", **kwargs)
    plt.close(fig)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def mean_macro(runs: dict[str, dict[int, dict[str, pd.DataFrame]]], arm: str) -> pd.DataFrame:
    frames = []
    for seed, data in sorted(runs[arm].items()):
        frame = data["macro"].copy()
        frame["seed"] = seed
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in panel.columns if c not in {"seed"}]
    return panel[numeric_cols].groupby("day", as_index=False).mean(numeric_only=True).sort_values("day")


def mean_insurer_daily(runs: dict[str, dict[int, dict[str, pd.DataFrame]]]) -> pd.DataFrame:
    frames = []
    for seed, data in sorted(runs["on"].items()):
        ins = data["insurer"].copy()
        daily = ins.groupby("day").agg(
            capital=("capital", "sum"),
            capital_ratio=("capital_ratio", "mean"),
            premiums_today=("premiums_today", "sum"),
            claims_today=("claims_today", "sum"),
            active_policies=("active_policies", "sum"),
        )
        daily["capital_ratio_total"] = daily["capital"] / daily["capital"].iloc[0]
        daily["premium_inflow_pct"] = (
            daily["premiums_today"] / daily["capital"].replace(0, np.nan)
        ).fillna(0.0) * 100.0
        daily["seed"] = seed
        frames.append(daily.reset_index())
    panel = pd.concat(frames, ignore_index=True)
    return panel.groupby("day", as_index=False).mean(numeric_only=True).sort_values("day")


def concat_firms(
    runs: dict[str, dict[int, dict[str, pd.DataFrame]]], arm: str, *, active_only: bool = False
) -> pd.DataFrame:
    frames = []
    for seed, data in sorted(runs[arm].items()):
        frame = data["firm"].copy()
        frame["seed"] = seed
        if "panic_level" not in frame.columns:
            frame["panic_level"] = frame["panic"]
        if active_only:
            frame = frame[as_bool(frame["active"])]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def day_x(df: pd.DataFrame | pd.Series) -> pd.Series:
    values = df["day"] if isinstance(df, pd.DataFrame) else df
    return (values.astype(float) + 1.0) * DAYS_PER_ROUND


def setup_day_axis(ax) -> None:
    ax.set_xlim(0, CALENDAR_HORIZON_DAYS)
    ax.set_xticks(DAY_TICKS)


def prepend_origin(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    x_out = pd.concat([pd.Series([0.0]), x.reset_index(drop=True)], ignore_index=True)
    y_out = pd.concat([pd.Series([0.0]), y.reset_index(drop=True)], ignore_index=True)
    return x_out, y_out


def mean_active_share_by_category(
    runs: dict[str, dict[int, dict[str, pd.DataFrame]]],
    *,
    arm: str,
    category_col: str,
    categories: list[str],
    condition_col: str,
) -> pd.DataFrame:
    """Mean active-firm share by category across paired seeds."""
    frames = []
    for seed, data in sorted(runs[arm].items()):
        firm = data["firm"].copy()
        firm["active_b"] = as_bool(firm["active"])
        firm["condition_b"] = as_bool(firm[condition_col])
        days = sorted(firm["day"].unique())
        active_n = firm[firm["active_b"]].groupby("day")["firm_id"].nunique().reindex(days).fillna(0)
        out = pd.DataFrame({"day": days})
        for category in categories:
            count = (
                firm[firm["active_b"] & firm["condition_b"] & (firm[category_col].astype(str) == category)]
                .groupby("day")["firm_id"]
                .nunique()
                .reindex(days)
                .fillna(0)
            )
            out[category] = count.to_numpy() / active_n.clip(lower=1).to_numpy() * 100.0
        out["seed"] = seed
        frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    return panel.groupby("day", as_index=False)[categories].mean(numeric_only=True).sort_values("day")


def prepend_zero_frame(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    x = np.concatenate([[0.0], day_x(df).to_numpy()])
    series = [np.concatenate([[0.0], df[col].to_numpy(dtype=float)]) for col in columns]
    total = np.sum(np.vstack(series), axis=0)
    return x, series, total


def style_market_stack_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Day", fontsize=8.5, labelpad=1.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, labelpad=2.0)
    ax.tick_params(axis="both", labelsize=7.0, pad=1.5)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    setup_day_axis(ax)
    ax.set_xticks([0, 150, 300])


def figure_eval_market_stack(runs) -> None:
    industry_order = list(INDUSTRY_LABELS.keys())
    industry_short = list(INDUSTRY_LABELS.values())
    insurer_order = [
        "Insurer_Apex_Global",
        "Insurer_Digital_CN",
        "Insurer_Mutual_Commercial",
        "Insurer_Specialty_Tech",
    ]
    insurer_short = ["Diversified", "Digital", "Mutual", "Specialty"]
    vendor_short = ["Alpha", "Beta", "Gamma", "Delta"]
    vendor_colors = ["#4E6D8C", "#849DB3", "#AAA5B8", "#CBA8B0"]
    industry_colors = [
        "#91BED1",
        "#9DBDD8",
        "#A8C1D4",
        "#AAB8C8",
        "#98A8B7",
        "#A2A1B7",
        "#A7A4B2",
        "#B6A8B7",
        "#BDB3AD",
        "#C3AEAF",
        "#AA99A6",
    ]
    insurer_colors = ["#4E6D8C", "#8EB6A8", "#9FB8C8", "#BFA2B8"]
    total_color = "#233447"

    vendor_df = mean_active_share_by_category(
        runs,
        arm="on",
        category_col="vendor_id",
        categories=VENDOR_ORDER,
        condition_col="has_ai",
    )
    ai_industry_df = mean_active_share_by_category(
        runs,
        arm="on",
        category_col="industry",
        categories=industry_order,
        condition_col="has_ai",
    )
    insurer_df = mean_active_share_by_category(
        runs,
        arm="on",
        category_col="insurer_id",
        categories=insurer_order,
        condition_col="has_insurance",
    )
    ins_industry_df = mean_active_share_by_category(
        runs,
        arm="on",
        category_col="industry",
        categories=industry_order,
        condition_col="has_insurance",
    )

    fig = plt.figure(figsize=(7.16, 1.72))
    outer = fig.add_gridspec(1, 4, left=0.06, right=0.995, bottom=0.24, top=0.85, wspace=0.24)
    axes = []
    legend_axes = []
    for idx in range(4):
        inner = outer[0, idx].subgridspec(1, 2, width_ratios=[1.0, 0.43], wspace=0.02)
        if idx == 0:
            ax = fig.add_subplot(inner[0, 0])
        else:
            ax = fig.add_subplot(inner[0, 0], sharex=axes[0], sharey=axes[0])
        legend_ax = fig.add_subplot(inner[0, 1])
        legend_ax.axis("off")
        axes.append(ax)
        legend_axes.append(legend_ax)
    panels = [
        (axes[0], vendor_df, VENDOR_ORDER, vendor_colors, vendor_short, "AI adoption (%)", "(a)"),
        (axes[1], ai_industry_df, industry_order, industry_colors, industry_short, "", "(b)"),
        (axes[2], insurer_df, insurer_order, insurer_colors, insurer_short, "Insurance coverage (%)", "(c)"),
        (axes[3], ins_industry_df, industry_order, industry_colors, industry_short, "", "(d)"),
    ]

    for ax, legend_ax, (_, df, columns, colors, labels, ylabel, panel_label) in zip(axes, legend_axes, panels):
        x, series, total = prepend_zero_frame(df, columns)
        stacks = ax.stackplot(x, series, labels=labels, colors=colors, alpha=0.94, linewidth=0.25, edgecolor="#F1F1F1")
        total_line, = ax.plot(x, total, color=total_color, linewidth=1.35, label="Total")
        style_market_stack_axis(ax, ylabel)
        ax.set_box_aspect(1.0)
        ax.text(0.0, 1.035, panel_label, transform=ax.transAxes, fontsize=9.0, fontweight="bold", va="bottom", ha="left")
        ax.tick_params(labelleft=panel_label in {"(a)", "(c)"})
        legend_ax.legend(
            list(stacks) + [total_line],
            labels + ["Total"],
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            frameon=False,
            fontsize=5.15 if panel_label in {"(b)", "(d)"} else 5.7,
            handlelength=0.85,
            handletextpad=0.35,
            labelspacing=0.12 if panel_label in {"(b)", "(d)"} else 0.24,
            borderaxespad=0.0,
        )
    save(fig, "fig_eval_market_stack", dpi=300)


def load_rule_abm_macro() -> dict[int, pd.DataFrame]:
    data: dict[int, pd.DataFrame] = {}
    missing: list[Path] = []
    for seed, name in RULE_ABM_RUN_NAMES.items():
        path = RULE_ABM_RUNS_DIR / name / "macro_daily.csv"
        if not path.exists():
            missing.append(path)
            continue
        data[seed] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError(
            "Missing rule-based ABM run outputs. Run the matched rule_heuristic baselines first:\n"
            + "\n".join(str(path) for path in missing)
        )
    return data


def figure_rule_abm_comparison(runs) -> None:
    llm_frames = {seed: data["macro"].copy() for seed, data in sorted(runs["on"].items())}
    rule_frames = load_rule_abm_macro()

    def panel_stats(frames_by_seed: dict[int, pd.DataFrame], metric: str) -> pd.DataFrame:
        rows = []
        for seed, frame in sorted(frames_by_seed.items()):
            df = frame.sort_values("day").copy()
            if metric == "adoption":
                values = df["ai_penetration"].fillna(0.0) * 100.0
            elif metric == "coverage":
                values = df["insurance_coverage_overall"].fillna(0.0) * 100.0
            elif metric == "uninsured_events":
                values = df["num_uninsured_claimable_events"].fillna(0.0).cumsum()
            elif metric == "bankruptcies":
                values = df["cumulative_bankruptcies"].fillna(0.0)
            else:  # pragma: no cover
                raise ValueError(metric)
            rows.append(pd.DataFrame({"day": df["day"].astype(float), "seed": seed, "value": values.astype(float)}))
        panel = pd.concat(rows, ignore_index=True)
        stats = (
            panel.groupby("day", as_index=False)["value"]
            .agg(mean="mean", low="min", high="max")
            .sort_values("day")
        )
        origin = pd.DataFrame({"day": [-1.0], "mean": [0.0], "low": [0.0], "high": [0.0]})
        return pd.concat([origin, stats], ignore_index=True)

    def x_from_stats(stats: pd.DataFrame) -> np.ndarray:
        return np.where(stats["day"].to_numpy(dtype=float) < 0.0, 0.0, (stats["day"].to_numpy(dtype=float) + 1.0) * DAYS_PER_ROUND)

    def draw_metric(ax, metric: str, ylabel: str, ylim: tuple[float, float] | None, panel_label: str) -> None:
        rule = panel_stats(rule_frames, metric)
        llm = panel_stats(llm_frames, metric)
        for stats, color, label, lw in [
            (rule, "#A97878", "Rule-based ABM", 2.1),
            (llm, "#6F7E8D", "Ours", 2.3),
        ]:
            x = x_from_stats(stats)
            mean = stats["mean"].to_numpy(dtype=float)
            low = stats["low"].to_numpy(dtype=float)
            high = stats["high"].to_numpy(dtype=float)
            ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=lw, label=label)
        setup_day_axis(ax)
        ax.set_xticks([0, 150, 300])
        ax.set_ylabel(ylabel, fontsize=7.8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymax = max(rule["high"].max(), llm["high"].max())
            ax.set_ylim(0, max(1.0, ymax * 1.18))
        ax.tick_params(labelsize=7.2, pad=1.5)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.text(0.04, 0.92, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=8.6, fontweight="bold")

    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.65), sharex=False)
    draw_metric(axes[0], "adoption", "AI Adoption (%)", (0, 100), "(a)")
    draw_metric(axes[1], "coverage", "Insurance (%)", (0, 100), "(b)")
    draw_metric(axes[2], "uninsured_events", "Uninsured Events", None, "(c)")
    draw_metric(axes[3], "bankruptcies", "Bankruptcies", None, "(d)")
    fig.supxlabel("Day", fontsize=8.2, y=0.02)
    handles = [
        Line2D([0], [0], color="#A97878", linewidth=2.5, label="Rule-based ABM"),
        Line2D([0], [0], color="#6F7E8D", linewidth=2.8, label="Ours"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.06), ncol=2, frameon=False, fontsize=8.2)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.84, wspace=0.42)
    save(fig, "4.3.1", dpi=320)


def figure_511_bankruptcy(runs) -> None:
    on = mean_macro(runs, "on")
    off = mean_macro(runs, "off")

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x_on = day_x(on)
    x_off = day_x(off)

    for _, data in sorted(runs["on"].items()):
        macro = data["macro"].sort_values("day")
        ax.plot(
            day_x(macro),
            macro["cumulative_bankruptcies"],
            color=BLUE,
            linewidth=0.9,
            alpha=0.16,
            drawstyle="steps-post",
            zorder=1,
        )
    for _, data in sorted(runs["off"].items()):
        macro = data["macro"].sort_values("day")
        ax.plot(
            day_x(macro),
            macro["cumulative_bankruptcies"],
            color=GREY,
            linewidth=0.9,
            alpha=0.18,
            linestyle="--",
            drawstyle="steps-post",
            zorder=1,
        )

    ax.plot(
        x_on,
        on["cumulative_bankruptcies"],
        color=BLUE,
        linewidth=2.3,
        drawstyle="steps-post",
        label="With Insurance mean",
        zorder=3,
    )
    ax.plot(
        x_off,
        off["cumulative_bankruptcies"],
        color=GREY,
        linewidth=2.3,
        linestyle="--",
        drawstyle="steps-post",
        label="No Insurance mean",
        zorder=3,
    )

    on_final = float(on["cumulative_bankruptcies"].iloc[-1])
    off_final = float(off["cumulative_bankruptcies"].iloc[-1])
    ax.text(CALENDAR_HORIZON_DAYS - 21, on_final + 0.12, f"{on_final:.2f}", color=BLUE, fontsize=12, ha="right")
    ax.text(CALENDAR_HORIZON_DAYS - 21, off_final + 0.12, f"{off_final:.2f}", color=GREY, fontsize=12, ha="right")

    ax.set_xlabel("Day", fontsize=20)
    ax.set_ylabel("Mean cumulative count", fontsize=18)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_ylim(0, 5.2)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    setup_day_axis(ax)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    fig.tight_layout()
    save(fig, "5.1.1")


def figure_512_social_capital(runs) -> None:
    on = mean_macro(runs, "on")
    off = mean_macro(runs, "off")

    fig = plt.figure(figsize=(8.6, 3.6))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.00],
        height_ratios=[1.0, 1.0],
        wspace=0.56,
        hspace=0.46,
    )
    ax = fig.add_subplot(gs[:, 0])
    color_solid = "#94B5D7"
    color_dashed = "#929EAB"
    color_gain_text = "#DC6C6E"

    on_total = on["social_total_capital"]
    off_total = off["social_total_capital"]
    ratio = on_total / off_total
    on_unabsorbed = on["unabsorbed_claimable_loss"].cumsum()
    off_unabsorbed = off["unabsorbed_claimable_loss"].cumsum()
    loss_reduction = (off_unabsorbed - on_unabsorbed) / 1e6
    ratio.iloc[0] = 1.0

    x = day_x(on)
    ax.plot(x, ratio, color=color_solid, linewidth=2.0, label="System-capital ratio")
    ax.set_xlabel("Day", fontsize=16)
    ax.set_ylabel("Insurance/no-insurance capital ratio", color="black", fontsize=13.5)
    ax.tick_params(axis="both", labelsize=13)
    setup_day_axis(ax)
    ax.set_ylim(0.988, 1.023)
    ax.set_yticks([0.99, 1.00, 1.01, 1.02])
    ax.set_yticklabels(["0.99", "1.00", "1.01", "1.02"])

    ax2 = ax.twinx()
    ax2.plot(
        x,
        loss_reduction,
        color=color_dashed,
        linewidth=2.0,
        linestyle="--",
        label="Unabsorbed loss reduction",
    )
    ax2.set_ylabel("Loss reduction (M)", color="black", fontsize=14, labelpad=1)
    ax2.set_ylim(-0.50, 2.30)
    ax2.set_yticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
    ax2.set_yticklabels(["-0.5", "0", "0.5", "1.0", "1.5", "2.0"])
    ax2.tick_params(axis="y", labelcolor="black", labelsize=12, pad=2)
    lines = [line for line in ax.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left", fontsize=10.4)

    labels = ["No Ins.", "With Ins."]
    values = [
        float(off_total.iloc[-1] / off_total.iloc[0] * 100.0),
        float(on_total.iloc[-1] / on_total.iloc[0] * 100.0),
    ]
    gain = values[1] - values[0]
    xpos = list(range(2))

    ax_in = fig.add_subplot(gs[0, 1])
    bars = ax_in.bar(xpos, values, color=[color_dashed, color_solid], width=0.5)
    for bar, val in zip(bars, values):
        ax_in.text(
            bar.get_x() + bar.get_width() / 2,
            val - 8.0,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=10.5,
            color="black",
            fontweight="bold",
        )
    ax_in.text(
        0.5,
        min(values) - 19.0,
        f"{gain:+.2f} pts",
        ha="center",
        va="center",
        fontsize=10,
        color=color_gain_text,
        fontweight="bold",
    )
    ax_in.set_ylim(0, 112)
    ax_in.set_title("Aggregate system capital", fontsize=10, pad=2)
    ax_in.set_yticks([0, 50, 100])
    ax_in.set_yticklabels(["0", "50", "100"])
    ax_in.set_xticks(xpos)
    ax_in.set_xticklabels([])
    ax_in.tick_params(axis="both", labelsize=10)

    ex_values = [
        float(off["unabsorbed_claimable_loss"].sum() / 1e6),
        float(on["unabsorbed_claimable_loss"].sum() / 1e6),
    ]
    ex_gain = ex_values[1] - ex_values[0]
    ax_in2 = fig.add_subplot(gs[1, 1])
    bars2 = ax_in2.bar(xpos, ex_values, color=[color_dashed, color_solid], width=0.5)
    for bar, val in zip(bars2, ex_values):
        ax_in2.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.30,
            f"{val:.2f}M",
            ha="center",
            va="top",
            fontsize=10.5,
            color="black",
            fontweight="bold",
        )
    ax_in2.text(
        0.5,
        max(ex_values) + 0.46,
        f"{abs(ex_gain):.2f}M less",
        ha="center",
        va="center",
        fontsize=10,
        color=color_gain_text,
        fontweight="bold",
    )
    ax_in2.set_ylim(0, max(ex_values) + 0.95)
    ax_in2.set_title("Unabsorbed claimable loss", fontsize=8.8, pad=2)
    ax_in2.set_yticks([0, 2, 4])
    ax_in2.set_yticklabels(["0", "2", "4"])
    ax_in2.set_xticks(xpos)
    ax_in2.set_xticklabels(labels, fontsize=7.8)
    ax_in2.tick_params(axis="both", labelsize=10)

    save(fig, "5.1.2")


def figure_513_adoption_coverage(runs) -> None:
    on = mean_macro(runs, "on")
    off = mean_macro(runs, "off")

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    on_ai_x, on_ai_y = prepend_origin(day_x(on), on["ai_penetration"])
    off_ai_x, off_ai_y = prepend_origin(day_x(off), off["ai_penetration"])
    on_ins_x, on_ins_y = prepend_origin(day_x(on), on["insurance_coverage_ai_adopters"])
    off_ins_x, off_ins_y = prepend_origin(day_x(off), off["insurance_coverage_ai_adopters"])

    ax.plot(on_ai_x, on_ai_y, color=TEAL, linewidth=2.0, label="AI Adoption, With Insurance")
    ax.plot(off_ai_x, off_ai_y, color=TEAL, linewidth=2.0, linestyle="--", label="AI Adoption, No Insurance")
    ax.plot(on_ins_x, on_ins_y, color=SAND, linewidth=2.0, label="Insurance Coverage, With Insurance")
    ax.plot(off_ins_x, off_ins_y, color=SAND, linewidth=2.0, linestyle="--", label="Insurance Coverage, No Insurance")

    ax.set_xlabel("Day", fontsize=20)
    ax.set_ylabel("Adoption / Coverage (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.margins(x=0, y=0)
    setup_day_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        frameon=False,
        fontsize=11,
        ncol=2,
        columnspacing=1.5,
        handlelength=2.8,
    )
    fig.subplots_adjust(bottom=0.34, left=0.12, right=0.98, top=0.96)
    save(fig, "5.1.3")


def figure_514_contract_terms(runs) -> tuple[float, float]:
    records = []
    for arm, label in [("on", "With Insurance"), ("off", "No Insurance")]:
        for seed, data in sorted(runs[arm].items()):
            events = data["events"]
            bound = events[events["event_type"].eq("vendor_contract_bound")]
            for value in pd.to_numeric(bound["term_days"], errors="coerce").dropna():
                records.append({"arm": label, "term": float(value), "seed": seed})
    terms = pd.DataFrame(records)
    on_mean = float(terms.loc[terms["arm"] == "With Insurance", "term"].mean())
    off_mean = float(terms.loc[terms["arm"] == "No Insurance", "term"].mean())

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    data = [
        terms.loc[terms["arm"] == "With Insurance", "term"],
        terms.loc[terms["arm"] == "No Insurance", "term"],
    ]
    parts = ax.violinplot(data, showmeans=False, showextrema=False)
    colors = ["#94B5D7", "#C7B1B2"]
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_edgecolor("#555555")
        pc.set_linewidth(1.1)
        pc.set_alpha(0.78)

    rng = np.random.default_rng(0)
    for i, series in enumerate(data, start=1):
        sample = series.dropna().astype(float)
        if len(sample) > 900:
            sample = sample.sample(900, random_state=0)
        jitter = rng.normal(0, 0.035, size=len(sample))
        ax.scatter(
            np.full(len(sample), i) + jitter,
            sample,
            s=7,
            color="black",
            alpha=0.13,
            linewidths=0,
            rasterized=True,
        )

    ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.16,
        showfliers=False,
        patch_artist=False,
        boxprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
    )
    means = []
    for series in data:
        sample = series.dropna().astype(float)
        means.append(float(sample.mean()) if len(sample) else 0.0)

    label_specs = {
        1: {"dx": 0.26, "dy": 7.2, "ha": "left"},
        2: {"dx": -0.26, "dy": 7.2, "ha": "right"},
    }
    for i, mean_value in enumerate(means, start=1):
        ax.scatter(
            [i],
            [mean_value],
            s=24,
            color=RED,
            edgecolor="white",
            linewidth=0.6,
            zorder=7,
        )
        spec = label_specs[i]
        text_x = i + spec["dx"]
        text_y = min(mean_value + spec["dy"], 123.0)
        ax.annotate(
            f"mean {mean_value:.1f}",
            xy=(i, mean_value),
            xytext=(text_x, text_y),
            textcoords="data",
            color="black",
            fontsize=11,
            va="bottom",
            ha=spec["ha"],
            arrowprops=dict(
                arrowstyle="-",
                color="#555555",
                linewidth=0.7,
                shrinkA=2,
                shrinkB=2,
            ),
            zorder=8,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["With Insurance", "No Insurance"], fontsize=14)
    ax.set_ylabel("Contract Term", fontsize=14)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylim(10, 128)
    ax.tick_params(axis="y", labelsize=14)
    fig.tight_layout()
    save(fig, "5.1.4")
    return off_mean, on_mean


def industry_iqr_series(runs, arm: str) -> pd.Series:
    by_seed = []
    for seed, data in sorted(runs[arm].items()):
        f = data["firm"].copy()
        f = f[as_bool(f["active"])]
        f["has_ai_b"] = as_bool(f["has_ai"])
        rows = []
        for day, gday in f.groupby("day"):
            rates = gday.groupby("industry")["has_ai_b"].mean() * 100.0
            rows.append({"day": day, seed: float(rates.quantile(0.75) - rates.quantile(0.25))})
        by_seed.append(pd.DataFrame(rows).set_index("day"))
    panel = pd.concat(by_seed, axis=1)
    return panel.mean(axis=1).sort_index()


def figure_611_diffusion_iqr(runs) -> None:
    on = mean_macro(runs, "on")
    off = mean_macro(runs, "off")
    on_iqr = industry_iqr_series(runs, "on")
    off_iqr = industry_iqr_series(runs, "off")

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(day_x(on), on["ai_penetration"] * 100, color="#384ADD", linewidth=2.0, label="With Insurance Adoption")
    ax.plot(day_x(off), off["ai_penetration"] * 100, color="#929EAB", linewidth=2.0, linestyle="--", label="No Insurance Adoption")
    ax.plot((on_iqr.index.astype(float) + 1.0) * DAYS_PER_ROUND, on_iqr.values, color="#384ADD", linewidth=1.6, alpha=0.55, label="With Insurance IQR")
    ax.plot((off_iqr.index.astype(float) + 1.0) * DAYS_PER_ROUND, off_iqr.values, color="#929EAB", linewidth=1.6, linestyle="--", alpha=0.65, label="No Insurance IQR")
    ax.set_xlabel("Day", fontsize=18)
    ax.set_ylabel("Adoption / industry IQR (%)", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    setup_day_axis(ax)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.tight_layout()
    save(fig, "6.1.1")


def figure_531_insurance_cycle(runs) -> None:
    df = mean_insurer_daily(runs)
    x = day_x(df)
    cap_ratio = df["capital_ratio_total"]
    premium_inflow_pct = df["premium_inflow_pct"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    blue = "#407993"
    red = "#E8C6B5"
    red_smooth = "#C77D73"

    ax.plot(x, cap_ratio, color=blue, linewidth=1.8, label="Insurer Capital Index")
    ax.fill_between(x, 0, cap_ratio, color=blue, alpha=0.15)
    ax.set_xlabel("Day", fontsize=20)
    ax.set_ylabel("Insurer Capital Index", color="black", fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelcolor="black", labelsize=20)
    setup_day_axis(ax)
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0)

    def smooth(values, window=9, sigma=2.0):
        values = np.asarray(values, dtype=float)
        if window % 2 == 0:
            window += 1
        half = window // 2
        xw = np.arange(-half, half + 1)
        kernel = np.exp(-(xw**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        padded = np.pad(values, (half, half), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    ax2 = ax.twinx()
    premium_smooth = smooth(premium_inflow_pct.to_numpy())
    ax2.plot(
        x,
        premium_inflow_pct,
        color=red,
        linestyle=(0, (2, 2)),
        linewidth=1.4,
        label="Premium Inflow / Capital",
    )
    ax2.plot(
        x,
        premium_smooth,
        color=red_smooth,
        linestyle="--",
        linewidth=1.2,
        label="Premium Inflow / Capital (Smoothed)",
    )
    ax2.set_ylabel("Premium inflow / capital (%)", color="black", fontsize=20)
    ax2.tick_params(axis="y", labelcolor="black", labelsize=20)
    ax2.set_ylim(bottom=0)

    lines = [line for line in ax.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    ax.legend(lines, [line.get_label() for line in lines], loc=(0.15, 0.75), frameon=False, fontsize=13)
    fig.tight_layout()
    save(fig, "5.3.1")


def figure_532_demand_supply(runs) -> None:
    df = mean_macro(runs, "on")
    day = day_x(df)
    cumulative_premiums = df["total_premiums"].fillna(0.0).cumsum() / 1e6
    cumulative_claims = df["total_claims"].fillna(0.0).cumsum() / 1e6
    daily_claims = df["total_claims"].fillna(0.0) / 1e6

    fig, ax = plt.subplots(figsize=(8.6, 3.25))
    premium_color = "#355C7D"
    claim_color = "#B5658D"
    buffer_color = "#C9D8E7"
    bar_color = "#DDB7CF"

    ax_bar = ax.twinx()
    ax_bar.bar(day, daily_claims, width=0.85 * DAYS_PER_ROUND, color=bar_color, alpha=0.38, edgecolor="none", label="Per-update claims")
    ax_bar.set_ylabel("Claims per update (M)", fontsize=14)
    daily_axis_top = max(float(daily_claims.max()) * 1.55, 0.1)
    ax_bar.set_ylim(0, daily_axis_top)
    ax_bar.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    ax_bar.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bar.tick_params(axis="y", labelsize=13)

    ax.set_zorder(ax_bar.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.fill_between(
        day,
        cumulative_claims,
        cumulative_premiums,
        where=(cumulative_premiums >= cumulative_claims),
        color=buffer_color,
        alpha=0.55,
        linewidth=0,
        label="Premium-minus-claim balance",
    )
    ax.plot(day, cumulative_premiums, color=premium_color, linewidth=2.4, label="Cumulative premiums")
    ax.plot(day, cumulative_claims, color=claim_color, linewidth=2.2, linestyle="--", label="Cumulative paid claims")

    top_claims = daily_claims.sort_values(ascending=False).head(3)
    for d, value in top_claims.items():
        if value <= 0:
            continue
        ax_bar.scatter([day.iloc[int(d)]], [float(value)], s=18, color=claim_color, zorder=4)

    final_day = float(day.iloc[-1])
    final_premium = float(cumulative_premiums.iloc[-1])
    final_claim = float(cumulative_claims.iloc[-1])
    ax.scatter([final_day], [final_premium], s=20, color=premium_color, zorder=5)
    ax.scatter([final_day], [final_claim], s=20, color=claim_color, zorder=5)

    ax.set_ylabel("Cumulative flow (M)", fontsize=16)
    ax.set_xlabel("Day", fontsize=18)
    setup_day_axis(ax)
    ax.set_ylim(0, max(final_premium, final_claim) * 1.32)
    ax.tick_params(axis="both", labelsize=13)

    handles = [
        Line2D([0], [0], color=premium_color, linewidth=2.8),
        Line2D([0], [0], color=claim_color, linewidth=2.5, linestyle="--"),
        matplotlib.patches.Patch(facecolor=buffer_color, edgecolor="none", alpha=0.55),
        matplotlib.patches.Patch(facecolor=bar_color, edgecolor="none", alpha=0.38),
    ]
    labels = [
        "Premiums\n(cum.)",
        "Claims\n(cum.)",
        "Buffer",
        "Per-update\nclaims",
    ]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.78, 0.56),
        frameon=False,
        fontsize=17.0,
        handlelength=1.55,
        labelspacing=1.05,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.11, right=0.68, bottom=0.19, top=0.95)
    save(fig, "5.3.2")


def figure_533_vendor_metrics(runs) -> None:
    df = concat_firms(runs, "on", active_only=True)
    last_day = int(df["day"].max())
    last = df[(df["day"] == last_day) & df["vendor_id"].notna()].copy()
    share_counts = last["vendor_id"].astype(str).value_counts()
    total = max(float(share_counts.sum()), 1.0)
    shares = [float(share_counts.get(v, 0)) / total for v in VENDOR_ORDER]

    vendor_df = df[df["vendor_id"].isin(VENDOR_ORDER)].copy()
    tail_risk = (
        vendor_df.groupby("vendor_id")["action_max_risk_score"].mean().reindex(VENDOR_ORDER).fillna(0.0)
        * pd.Series(VENDOR_RISK).reindex(VENDOR_ORDER)
    )
    avg_error = vendor_df.groupby("vendor_id")["task_failure_rate"].mean().reindex(VENDOR_ORDER).fillna(0.0)

    y = list(range(len(VENDOR_ORDER)))
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.2), sharey=True)
    titles = ["Market Share", "Tail Risk", "Avg. Error"]
    data_series = [shares, tail_risk.values, avg_error.values]
    gradients = [
        ["#dce7eb", "#ccd9e7", "#add4e5", "#accbdf"],
        ["#e9e5e6", "#e8d3d2", "#e7c5c4", "#ebbcb9"],
        ["#e4e9e3", "#dae4d9", "#cbe0d1", "#bbd4bf"],
    ]
    vendor_labels = [VENDOR_LABELS[v] for v in VENDOR_ORDER]

    for idx, (ax, values, title, grad) in enumerate(zip(axes, data_series, titles, gradients)):
        for yi, val, color in zip(y, values, grad):
            ax.barh(yi, val, height=0.42, color=color, edgecolor=color, linewidth=0.8)
        ax.set_xlabel(title, fontsize=20)
        ax.set_ylim(-0.5, len(y) - 0.5)
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, length=3, labelsize=20)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, length=0, labelsize=10)
        ax.minorticks_off()
        if idx == 0:
            ax.set_ylabel("Vendor", fontsize=20)
            ax.set_yticks(y)
            ax.set_yticklabels(vendor_labels)
            ax.tick_params(axis="y", labelleft=True, labelsize=17)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.margins(y=0.02)
    fig.tight_layout()
    save(fig, "5.3.3")


def abbr_industry(name: str) -> str:
    key = str(name)
    if key in INDUSTRY_LABELS:
        return INDUSTRY_LABELS[key]
    parts = key.replace("-", "_").split("_")
    if len(parts) > 1:
        return "".join(p[:2].upper() for p in parts if p)
    return key[:4].upper()


def figure_534_ridgeline(runs) -> None:
    on = mean_macro(runs, "on")
    off = mean_macro(runs, "off")
    day = day_x(on)

    scale = 1000.0
    on_p95 = on["panic_p95"].fillna(0.0) * scale
    off_p95 = off["panic_p95"].fillna(0.0) * scale
    on_burden = on["avg_panic"].fillna(0.0).cumsum() * scale
    off_burden = off["avg_panic"].fillna(0.0).cumsum() * scale

    fig, (ax_tail, ax_burden) = plt.subplots(
        2,
        1,
        figsize=(8.6, 4.25),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.08},
    )

    ax_tail.fill_between(day, 0, off_p95, color=RED, alpha=0.18, linewidth=0)
    ax_tail.plot(day, off_p95, color=RED, linewidth=2.0, linestyle="--", label="No Insurance p95")
    ax_tail.fill_between(day, 0, on_p95, color=BLUE, alpha=0.13, linewidth=0)
    ax_tail.plot(day, on_p95, color=BLUE, linewidth=2.0, label="With Insurance p95")

    peak_off_day = int(day.iloc[int(off_p95.to_numpy().argmax())])
    peak_on_day = int(day.iloc[int(on_p95.to_numpy().argmax())])
    ax_tail.scatter([peak_off_day], [float(off_p95.max())], color=RED, s=30, zorder=5)
    ax_tail.scatter([peak_on_day], [float(on_p95.max())], color=BLUE, s=30, zorder=5)
    ax_tail.text(peak_off_day, float(off_p95.max()) + 1.2, f"{float(off_p95.max()):.1f}", ha="center", fontsize=10, color=RED)
    ax_tail.text(peak_on_day, float(on_p95.max()) + 1.0, f"{float(on_p95.max()):.1f}", ha="center", fontsize=10, color=BLUE)
    ax_tail.set_ylabel("p95 panic\n($\\times 10^3$)", fontsize=15)
    ax_tail.tick_params(axis="both", labelsize=13)
    ax_tail.set_ylim(0, max(float(off_p95.max()), float(on_p95.max())) * 1.22)
    ax_tail.legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2)

    ax_burden.fill_between(day, on_burden, off_burden, where=(off_burden >= on_burden), color="#D98B8C", alpha=0.30, linewidth=0)
    ax_burden.plot(day, off_burden, color=RED, linewidth=2.0, linestyle="--", label="No Insurance burden")
    ax_burden.plot(day, on_burden, color=BLUE, linewidth=2.0, label="With Insurance burden")
    ax_burden.scatter([float(day.iloc[-1])], [float(off_burden.iloc[-1])], color=RED, s=26, zorder=5)
    ax_burden.scatter([float(day.iloc[-1])], [float(on_burden.iloc[-1])], color=BLUE, s=26, zorder=5)
    ax_burden.text(float(day.iloc[-1]) - 6.0, float(off_burden.iloc[-1]) + 7.0, f"{float(off_burden.iloc[-1]):.1f}", ha="right", fontsize=10, color=RED)
    ax_burden.text(float(day.iloc[-1]) - 6.0, float(on_burden.iloc[-1]) + 7.0, f"{float(on_burden.iloc[-1]):.1f}", ha="right", fontsize=10, color=BLUE)
    ax_burden.set_ylabel("Mean panic\nburden ($\\times 10^3$)", fontsize=15)
    ax_burden.set_xlabel("Day", fontsize=17)
    ax_burden.tick_params(axis="both", labelsize=13)
    ax_burden.set_ylim(0, float(off_burden.max()) * 1.18)
    ax_burden.legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2)
    setup_day_axis(ax_burden)
    fig.align_ylabels([ax_tail, ax_burden])
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.98)
    save(fig, "5.3.4", dpi=300)


def figure_535_sentiment_stack(runs) -> None:
    def industry_corr(arm: str, min_day: int = 20) -> pd.DataFrame:
        df = concat_firms(runs, arm, active_only=True)
        mat = (
            df.groupby(["day", "industry"])["panic_level"]
            .mean()
            .unstack()
            .fillna(0.0)
            .sort_index()
        )
        mat = mat[mat.index >= min_day]
        return mat.corr().fillna(0.0)

    corr_on = industry_corr("on")
    corr_off = industry_corr("off")

    off_df = concat_firms(runs, "off", active_only=True)
    order = (
        off_df[off_df["day"] >= 20]
        .groupby("industry")["panic_level"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    corr_on = corr_on.reindex(index=order, columns=order).fillna(0.0)
    corr_off = corr_off.reindex(index=order, columns=order).fillna(0.0)

    labels = [abbr_industry(ind) for ind in order]
    n = len(order)
    cmap = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=55, ha="left", rotation_mode="anchor", fontsize=18)
    ax.set_yticklabels(labels, fontsize=18)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=2)
    ax.tick_params(axis="y", left=False, length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#D7D7D7", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            if i == j:
                val = 1.0
                radius = 0.36
                face = cmap(norm(val))
            elif j > i:
                val = float(corr_on.iloc[i, j])
                radius = 0.39 * math.sqrt(abs(val))
                face = cmap(norm(val))
            else:
                val = float(corr_off.iloc[i, j])
                radius = 0.39 * math.sqrt(abs(val))
                face = cmap(norm(val))
            if i != j and abs(val) < 0.015:
                continue
            circle = matplotlib.patches.Circle(
                (j, i),
                radius,
                facecolor=face,
                edgecolor="#777777",
                linewidth=0.35,
                alpha=0.92,
            )
            ax.add_patch(circle)

    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("Correlation", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.06, top=0.86)
    save(fig, "5.3.5")


def figure_536_sector_vendor_polar(runs) -> None:
    df = concat_firms(runs, "on", active_only=True)
    day_df = df[df["day"] == df["day"].max()].copy()
    day_df["vendor_id"] = day_df["vendor_id"].astype(str)
    day_df["has_insurance"] = as_bool(day_df["has_insurance"])

    industry_counts = day_df.groupby("industry").size().sort_values(ascending=False)
    industries = industry_counts.index.tolist()
    insured_counts = day_df[day_df["has_insurance"]].groupby("industry").size().reindex(industries, fill_value=0)
    industry_total = industry_counts.reindex(industries).fillna(0)
    insured_share = (insured_counts / industry_total.replace(0, np.nan)).fillna(0.0)
    vendor_counts = (
        day_df[day_df["vendor_id"].isin(VENDOR_ORDER)]
        .groupby(["industry", "vendor_id"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=industries, columns=VENDOR_ORDER, fill_value=0)
    )

    total_ind = industry_total.sum() if industry_total.sum() > 0 else 1.0
    weights = (industry_total / total_ind).to_numpy()
    widths = weights * (2 * np.pi)

    blue_row = ["#dce7eb", "#ccd9e7", "#add4e5", "#accbdf", "#8cb8d3", "#83b7cf"]
    pink_row = ["#e9e5e6", "#e8d3d2", "#e7c5c4", "#ebbcb9", "#deb4b5", "#c89a9c"]
    green_row = ["#cbe0d1", "#afcdb1", "#8fb69a", "#7fab87", "#6f9b7c", "#5f8f70"]
    industry_colors = blue_row + pink_row[: max(0, len(industries) - len(blue_row))]
    vendor_colors = green_row[: len(VENDOR_ORDER)]
    uninsured_color = "#6B6B6B"

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw={"projection": "polar"})
    ax.set_axis_off()
    legend_handles = [
        matplotlib.patches.Patch(color=vendor_colors[0], label="Vendor Alpha"),
        matplotlib.patches.Patch(color=vendor_colors[1], label="Vendor Beta"),
        matplotlib.patches.Patch(color=vendor_colors[2], label="Vendor Gamma"),
        matplotlib.patches.Patch(color=vendor_colors[3], label="Vendor Delta"),
        matplotlib.patches.Patch(color=uninsured_color, label="Uninsured"),
    ]

    angle = 0.0
    gap = 0.07
    bar_bottom = 0.14
    desired_bar_top = 0.82
    vendor_totals = vendor_counts.sum(axis=1).replace(0, np.nan)
    vendor_shares = vendor_counts.div(vendor_totals, axis=0).fillna(0.0)
    max_frac = float(vendor_shares.to_numpy().max()) if not vendor_shares.empty else 0.0
    bar_scale = 0.50 if max_frac <= 0 else (desired_bar_top - bar_bottom) / max_frac
    tick_radii = [0.60, 0.80]
    inner_sector_bottom = 0.10
    inner_sector_top = 0.88
    inner_sector_height = inner_sector_top - inner_sector_bottom
    outer_ring_bottom = inner_sector_top + 0.06
    outer_ring_thick = 0.12
    vendor_letters = ["A", "B", "G", "D"]

    for idx, ind in enumerate(industries):
        width = max(0.01, widths[idx] - gap)
        sector_start = angle + gap / 2
        sector_end = angle + widths[idx] - gap / 2
        center = angle + widths[idx] / 2
        ind_color = industry_colors[idx % len(industry_colors)]

        ax.bar([center], [inner_sector_height], width=width, bottom=inner_sector_bottom, color=ind_color, edgecolor=ind_color, linewidth=0.0, align="center", alpha=0.65)
        ax.bar([center], [outer_ring_thick], width=width, bottom=outer_ring_bottom, color=ind_color, edgecolor=ind_color, linewidth=0.0, align="center")

        counts = vendor_counts.loc[ind]
        total = counts.sum() if counts.sum() > 0 else 1.0
        for vidx, vid in enumerate(VENDOR_ORDER):
            frac = float(counts.get(vid, 0)) / total
            bar_center = angle + (vidx + 0.5) * (width / len(VENDOR_ORDER))
            bar_height = bar_scale * frac
            ax.bar(
                [bar_center],
                [bar_height],
                width=width / len(VENDOR_ORDER) * 0.85,
                bottom=bar_bottom,
                color=vendor_colors[vidx],
                edgecolor=vendor_colors[vidx],
                linewidth=0.6,
                align="center",
            )
            ax.text(
                bar_center,
                bar_bottom + bar_height + 0.02,
                vendor_letters[vidx],
                ha="center",
                va="bottom",
                fontsize=13,
                color="#4A4A4A",
            )

        unins_share = 1.0 - float(insured_share.loc[ind])
        ring_thick = 0.08
        ring_height = ring_thick * unins_share
        ring_bottom = inner_sector_top - ring_height
        ax.bar([center], [ring_height], width=width * 0.98, bottom=ring_bottom, color=uninsured_color, edgecolor=uninsured_color, linewidth=0.0, align="center", zorder=5)

        arc_theta = np.linspace(sector_start, sector_end, 60)
        label_theta = sector_end - 0.02
        for r in tick_radii:
            ax.plot(arc_theta, [r] * len(arc_theta), color="#9A9A9A", lw=0.6, ls="--", alpha=0.4)
            ax.text(
                label_theta,
                r + 0.04,
                f"{r:.1f}",
                ha="left",
                va="center",
                rotation=np.degrees(label_theta) - 90,
                rotation_mode="anchor",
                fontsize=17,
                color="#7A7A7A",
            )
        ax.text(
            center,
            outer_ring_bottom + outer_ring_thick / 2,
            abbr_industry(ind),
            ha="center",
            va="center",
            rotation=np.degrees(center) - 90,
            rotation_mode="anchor",
            fontsize=24,
            color="#4A4A4A",
        )
        angle += widths[idx]

    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.30, 1.02), frameon=False, fontsize=22)
    fig.subplots_adjust(right=0.75)
    save(fig, "5.3.6", dpi=300)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    runs = load_runs()
    figure_eval_market_stack(runs)
    figure_rule_abm_comparison(runs)
    figure_511_bankruptcy(runs)
    figure_512_social_capital(runs)
    figure_513_adoption_coverage(runs)
    off_mean, on_mean = figure_514_contract_terms(runs)
    figure_611_diffusion_iqr(runs)
    figure_531_insurance_cycle(runs)
    figure_532_demand_supply(runs)
    figure_533_vendor_metrics(runs)
    figure_534_ridgeline(runs)
    figure_535_sentiment_stack(runs)
    figure_536_sector_vendor_polar(runs)
    print(f"Regenerated paper figures in {OUTPUT_DIR}")
    print(f"Contract term means: insurance off={off_mean:.1f} updates, on={on_mean:.1f} updates")


if __name__ == "__main__":
    main()

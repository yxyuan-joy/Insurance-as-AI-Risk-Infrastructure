import argparse
import os

import pandas as pd


FATAL_STDERR_PATTERN = "额度已用尽|Error during processing|401|400 This model|TIMEOUT|COMMAND_NOT_FOUND"


def drop_infra_failures(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "stderr" not in df.columns or df.empty:
        return df, 0
    stderr = df["stderr"].fillna("").astype(str)
    mask = stderr.str.contains(FATAL_STDERR_PATTERN, regex=True)
    return df.loc[~mask].copy(), int(mask.sum())


def episode_key_columns(df: pd.DataFrame) -> list[str]:
    if "task_id" in df.columns:
        return ["firm_id", "day", "task_id"]
    if "task_index" in df.columns:
        return ["firm_id", "day", "task_index"]
    return ["firm_id", "day"]


def main():
    parser = argparse.ArgumentParser(
        description="Merge a completed base benchmark csv with resumed benchmark output."
    )
    parser.add_argument("--base", required=True, help="Existing deduplicated benchmark csv.")
    parser.add_argument("--resume", required=True, help="New benchmark csv produced by resumed run.")
    parser.add_argument("--output", required=True, help="Merged output csv path.")
    parser.add_argument(
        "--keep-infra-failures",
        action="store_true",
        help="Keep rows where AutoCLAW failed because of quota/runtime infrastructure errors.",
    )
    args = parser.parse_args()

    base_df = pd.read_csv(args.base)
    resume_df = pd.read_csv(args.resume)
    dropped_base = dropped_resume = 0
    if not args.keep_infra_failures:
        base_df, dropped_base = drop_infra_failures(base_df)
        resume_df, dropped_resume = drop_infra_failures(resume_df)

    frames = [df for df in [base_df, resume_df] if len(df) > 0]
    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=base_df.columns)
    merged["_source_order"] = range(len(merged))
    key_cols = episode_key_columns(merged)
    merged = merged.sort_values(["firm_id", "day", "_source_order"])
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    merged = merged.sort_values(["day", "firm_id"]).drop(columns=["_source_order"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"base_rows={len(base_df)}")
    print(f"resume_rows={len(resume_df)}")
    print(f"dropped_infra_failures_base={dropped_base}")
    print(f"dropped_infra_failures_resume={dropped_resume}")
    print(f"merged_rows={len(merged)}")
    print(f"merged_firms={merged['firm_id'].nunique()}")
    print(f"day_min={merged['day'].min()} day_max={merged['day'].max()}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

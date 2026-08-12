import argparse
import glob
import os
from pathlib import Path

import pandas as pd


FATAL_STDERR_PATTERN = "额度已用尽|Error during processing|401|400 This model|TIMEOUT|COMMAND_NOT_FOUND"


def drop_infra_failures(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "stderr" not in df.columns or df.empty:
        return df, 0
    stderr = df["stderr"].fillna("").astype(str)
    mask = stderr.str.contains(FATAL_STDERR_PATTERN, regex=True)
    return df.loc[~mask].copy(), int(mask.sum())


def read_csvs(pattern: str) -> list[pd.DataFrame]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {pattern}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["_source_file"] = path
        frames.append(df)
    return frames


def episode_key_columns(df: pd.DataFrame) -> list[str]:
    if "task_id" in df.columns:
        return ["firm_id", "day", "task_id"]
    if "task_index" in df.columns:
        return ["firm_id", "day", "task_index"]
    return ["firm_id", "day"]


def drop_resolved_errors(errors: pd.DataFrame, merged: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if errors.empty or merged.empty:
        return errors, 0
    key_cols = [col for col in episode_key_columns(merged) if col in errors.columns and col in merged.columns]
    if not key_cols:
        return errors, 0
    success_keys = set()
    for row in merged[key_cols].itertuples(index=False, name=None):
        success_keys.add(tuple(str(value) if col != "day" else int(value) for col, value in zip(key_cols, row)))

    def is_resolved(row) -> bool:
        key = []
        for col in key_cols:
            value = row[col]
            key.append(str(value) if col != "day" else int(float(value)))
        return tuple(key) in success_keys

    mask = errors.apply(is_resolved, axis=1)
    return errors.loc[~mask].copy(), int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description="Merge multiple AutoCLAW benchmark shard outputs.")
    parser.add_argument("--input-glob", required=True, help="Glob for shard firm_daily_action_risk.csv files.")
    parser.add_argument("--output", required=True, help="Merged benchmark CSV path.")
    parser.add_argument("--error-glob", default="", help="Optional glob for shard benchmark_errors.csv files.")
    parser.add_argument("--error-output", default="", help="Optional merged error CSV path.")
    parser.add_argument(
        "--keep-infra-failures",
        action="store_true",
        help="Keep rows where AutoCLAW failed because of quota/runtime infrastructure errors.",
    )
    args = parser.parse_args()

    frames = read_csvs(args.input_glob)
    merged = pd.concat(frames, ignore_index=True)
    original_rows = len(merged)
    dropped_infra = 0
    if not args.keep_infra_failures:
        merged, dropped_infra = drop_infra_failures(merged)

    merged["_source_order"] = range(len(merged))
    merged = merged.sort_values(["firm_id", "day", "_source_order"])
    key_cols = episode_key_columns(merged)
    duplicate_rows = int(merged.duplicated(subset=key_cols).sum())
    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    merged = merged.sort_values(["day", "firm_id"]).drop(columns=["_source_order", "_source_file"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    print(f"input_files={len(frames)}")
    print(f"input_rows={original_rows}")
    print(f"dropped_infrastructure_failure_rows={dropped_infra}")
    print(f"duplicate_firm_day_rows={duplicate_rows}")
    print(f"merged_rows={len(merged)}")
    print(f"merged_firms={merged['firm_id'].nunique() if 'firm_id' in merged.columns else 0}")
    if "day" in merged.columns and len(merged) > 0:
        print(f"day_min={merged['day'].min()} day_max={merged['day'].max()}")
    print(f"output={output}")

    if args.error_glob and args.error_output:
        error_paths = sorted(glob.glob(args.error_glob))
        error_frames = [pd.read_csv(path).assign(_source_file=path) for path in error_paths if os.path.getsize(path) > 0]
        if error_frames:
            errors = pd.concat(error_frames, ignore_index=True)
            errors, resolved_errors = drop_resolved_errors(errors, merged)
        else:
            errors = pd.DataFrame()
            resolved_errors = 0
        error_output = Path(args.error_output)
        error_output.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(error_output, index=False)
        print(f"error_files={len(error_paths)}")
        print(f"resolved_error_rows_dropped={resolved_errors}")
        print(f"error_rows={len(errors)}")
        print(f"error_output={error_output}")


if __name__ == "__main__":
    main()

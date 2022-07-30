import argparse
from typing import List
import pandas as pd
import os
from tqdm import tqdm


def create_ensemble_submission(files: List[str], output: str) -> None:
    print(f"Files: {files}")

    concat_df = pd.read_csv(files[0], index_col=0)
    concat_df.rename({"prediction": "prediction_0"}, axis=1, inplace=True)

    for i, f in tqdm(enumerate(files[1:], 1)):
        df = pd.read_csv(f, index_col=0)
        concat_df = pd.concat([concat_df, df], axis=1)
        concat_df.rename({"prediction": f"prediction_{i}"}, axis=1, inplace=True)

    concat_df["avg"] = 0
    for col in concat_df.columns:
        if col != "avg":
            concat_df["avg"] += concat_df[col]
    concat_df["avg"] /= len(files)

    concat_df["ensemble"] = 0
    concat_df["ensemble"][concat_df["avg"] >= 0.5] = 1
    concat_df["ensemble"].value_counts()

    concat_df.rename({"ensemble": "prediction"}, axis=1, inplace=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    concat_df["prediction"].to_csv(output, index=True, header=True)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        metavar="N",
        type=str,
        nargs="+",
        help="Files to be added to the ensemble",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file",
        default="./ensemble.csv",
    )

    args = parser.parse_args()

    files = args.files

    for f in files:
        assert f.endswith(".csv"), f"File must be a csv, incorrect path: {f}"

    create_ensemble_submission(files, output=args.output)

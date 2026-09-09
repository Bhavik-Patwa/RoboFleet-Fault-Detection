
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    PROJECT_ROOT / "data" / "alfa" / "metadata"
    / "training_flight_reference.csv"
)

RECORDING_DATE_PATTERN = r"^carbonZ_(\d{4}-\d{2}-\d{2})-"


# Checking whether each fault family remains represented when a date is held out
def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH)

    required_columns = {
        "flight_name",
        "fault_family",
        "fault_classification_candidate"
    }

    missing_columns = required_columns.difference(reference.columns)

    if missing_columns:
        raise RuntimeError(f"Missing reference columns : {sorted(missing_columns)}")

    flights = reference.loc[
        reference["fault_classification_candidate"].eq(True)
    ].copy()

    if flights.empty or flights["flight_name"].duplicated().any():
        raise RuntimeError("Expected non-empty, unique fault-flight records.")

    flights["recording_date"] = flights["flight_name"].str.extract(
        RECORDING_DATE_PATTERN,
        expand = False
    )

    if flights[["recording_date", "fault_family"]].isna().any().any():
        raise RuntimeError("Fault flights have missing dates or fault families.")

    coverage = pd.crosstab(
        flights["fault_family"],
        flights["recording_date"]
    )

    print("Fault-flight counts by family and recording date :")
    print(coverage.to_string())

    # Identifying test classes that a supervised classifier cannot learn
    # because no examples remain in the corresponding training partition.
    print("\nFamilies absent from training when each date is held out :")

    for recording_date in coverage.columns:
        training_counts = coverage.drop(
            columns = recording_date
        ).sum(axis = 1)

        unseen_families = coverage.index[
            coverage[recording_date].gt(0)
            & training_counts.eq(0)
        ].tolist()

        print(
            f"{recording_date} : "
            f"{', '.join(unseen_families) if unseen_families else 'None'}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fault-family coverage check failed : {exc}")
        sys.exit(1)
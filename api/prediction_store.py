import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from contextlib import closing

import numpy as np
import pandas as pd


class PredictionStoreError(RuntimeError):
    pass


# Creating the durable prediction-event schema
def initialize_prediction_store(database_path: Path) -> None:
    try:
        database_path.parent.mkdir(
            parents = True,
            exist_ok = True
        )

        # Closing the connection explicitly after the transaction finishes
        with closing(
            sqlite3.connect(
                database_path,
                timeout = 30
            )
        ) as connection:
            with connection:
                connection.execute(
                    "PRAGMA journal_mode = WAL"
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        record_index INTEGER NOT NULL,
                        observed_at_utc TEXT NOT NULL,
                        model_uri TEXT NOT NULL,
                        fault_score REAL NOT NULL,
                        alert INTEGER NOT NULL CHECK (alert IN (0, 1)),
                        features_json TEXT NOT NULL,
                        UNIQUE (batch_id, record_index)
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    prediction_events_model_time_index
                    ON prediction_events (
                        model_uri,
                        observed_at_utc
                    )
                    """
                )

    except (OSError, sqlite3.Error) as exc:
        raise PredictionStoreError("Prediction store could not be initialized.") from exc


# Recording one accepted event for every prediction
def record_prediction_events(database_path: Path, model_uri: str,
                             features: pd.DataFrame, fault_scores: np.ndarray,
                             alerts: np.ndarray
) -> str:
    if len(features) == 0:
        raise PredictionStoreError("At least one prediction event is required.")

    if not (
        len(features)
        == len(fault_scores)
        == len(alerts)
    ):
        raise PredictionStoreError("Prediction event counts do not match.")

    batch_id = str(uuid4())
    observed_at_utc = datetime.now(
        timezone.utc
    ).isoformat()

    event_records = []

    for record_index, feature_values in enumerate(
        features.to_numpy(dtype = "float64")
    ):
        event_records.append(
            (
                batch_id,
                record_index,
                observed_at_utc,
                model_uri,
                float(fault_scores[record_index]),
                int(alerts[record_index]),
                json.dumps(
                    feature_values.tolist(),
                    separators = (",", ":"),
                    allow_nan = False
                )
            )
        )

    try:
        # Closing the connection explicitly after the atomic batch insert
        with closing(
            sqlite3.connect(
                database_path,
                timeout = 30
            )
        ) as connection:
            with connection:
                connection.executemany(
                    """
                    INSERT INTO prediction_events (
                        batch_id,
                        record_index,
                        observed_at_utc,
                        model_uri,
                        fault_score,
                        alert,
                        features_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_records
                )

    except (
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError
    ) as exc:
        raise PredictionStoreError("Prediction events could not be recorded.") from exc

    return batch_id
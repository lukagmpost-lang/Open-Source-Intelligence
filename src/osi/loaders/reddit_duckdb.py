"""One month of Arctic Shift comments, filtered in DuckDB before it reaches pandas.

The parquet lives at https://huggingface.co/datasets/open-index/arctic.
DuckDB reads it over HTTP. Nothing is written to a database file.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# Spill goes here when a month does not fit in the memory cap.
_SPILL = "/tmp/duckdb_spill"


def load_month(year: int, month: int, subreddits: list[str]) -> pd.DataFrame:
    """Comments from one calendar month, limited to the given subreddits.

    Columns are author, subreddit, and thread_id. Arctic stores the thread
    on link_id, so that column is renamed here.
    """
    # An empty path is an in-memory database. There is no file to reopen later.
    connection = duckdb.connect()
    try:
        # hf:// is an HTTP read. httpfs is the extension that speaks it.
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
        # A full month is a few gigabytes. Cap the process and spill the rest.
        connection.execute("SET memory_limit = '4GB'")
        # DuckDB does not create the spill folder. Make it before the scan.
        Path(_SPILL).mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = '{_SPILL}'")
        connection.execute("SET threads = 4")
        # Row order is not part of the result. Skipping it lets the scan stay parallel.
        connection.execute("SET preserve_insertion_order = false")
        # month:02d matches the dataset layout, for example 2012/08.
        path = (
            "hf://datasets/open-index/arctic/data/comments/"
            f"{int(year)}/{int(month):02d}/*.parquet"
        )
        frame = connection.execute(
            """
            SELECT author, subreddit, link_id AS thread_id
            FROM read_parquet(?)
            WHERE subreddit IN (SELECT UNNEST(?))
            """,
            [path, list(subreddits)],
        ).df()
    finally:
        connection.close()
    return frame

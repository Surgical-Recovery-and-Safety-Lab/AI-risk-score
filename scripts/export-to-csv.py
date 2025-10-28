#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export-to-csv.py

Exports a table from a duckdb to a csv file.
Author: Mathias Roesler
Date 10/25
"""

import duckdb
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a table from a duckdb to a csv file"
    )
    parser.add_argument(
        "duckdb_file",
        metavar="duckdb-file",
        help="Path to the duckdb file",
    )
    parser.add_argument(
        "table_name",
        metavar="table-name",
        help="Name of the table to export from the database",
    )
    parser.add_argument(
        "csv_path", metavar="csv-path", help="Path to the export csv file"
    )

    args = parser.parse_args()

    con = duckdb.connect(args.duckdb_file)  # Connect to the database
    query = f"SELECT * FROM {args.table_name}"

    df = con.execute(query).df()  # Get the data
    con.close()

    df.to_csv(args.csv_path, index=False)

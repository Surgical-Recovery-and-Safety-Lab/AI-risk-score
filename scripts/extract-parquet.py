#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract-parquet.py

Extracts the data from a parquet file and creates a duckdb file
Author: Mathias Roesler
Date 10/25
"""

import duckdb
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a parquet file to a duckdb file"
    )
    parser.add_argument(
        "parquet_file",
        metavar="parquet-file",
        help="Path to the parquet file",
    )
    parser.add_argument(
        "duckdb_file",
        metavar="duckdb-file",
        help="Path to the duckdb file",
    )
    parser.add_argument(
        "-t",
        "--table-name",
        default="default_table",
        help="Name of the table to create in the database",
    )

    args = parser.parse_args()

    con = duckdb.connect(args.duckdb_file)  # Connect to new database

    query = "CREATE TABLE {} AS SELECT * FROM read_parquet({})".format(
        args.table_name, args.parquet_file
    )

    # Execute the query
    con.execute(query)
    con.close()

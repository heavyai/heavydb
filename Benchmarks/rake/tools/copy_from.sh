#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import *.dat files into tables.
# Required: tables must already exist.

# These utilty scripts are intended to be customized.
# Customize these values:
DATABASE=tpcds_500gb
# Location of *.dat files
DAT_DIR="/var/lib/heavyai/tpc-ds/TPC-DS_Tools_v3.2.0_500GB"
HEAVYSQL="/opt/heavyai/bin/heavysql -p HyperInteractive $DATABASE"
TABLES="call_center catalog_page catalog_returns catalog_sales customer customer_address customer_demographics date_dim dbgen_version household_demographics income_band inventory item promotion reason ship_mode store store_returns store_sales time_dim warehouse web_page web_returns web_sales web_site"

# Set PARALLEL=1 if .dat files were not generated in PARALLEL.
PARALLEL=64

if (( 1 < PARALLEL ))
then
  for table in $TABLES
  do
    if [[ $table == 'customer' || $table == 'store' ]]
    then
      # QUERY must not match customer_address, customer_demographics, store_returns, or store_sales.
      # ? is not a valid wildcard for COPY FROM.
      for i in $(seq 1 $PARALLEL)
      do
        # There may only be 1 store table and so some of these might produce errors.
        QUERY="COPY $table FROM '$DAT_DIR/${table}_${i}_$PARALLEL.dat' WITH (DELIMITER='|', HEADER='false');"
        echo "$QUERY"
        echo "$QUERY" | $HEAVYSQL
      done
    else
      QUERY="COPY $table FROM '$DAT_DIR/${table}_*_$PARALLEL.dat' WITH (DELIMITER='|', HEADER='false');"
      echo "$QUERY"
      echo "$QUERY" | $HEAVYSQL
    fi
  done
else
  for table in $TABLES
  do
    QUERY="COPY $table FROM '$DAT_DIR/$table.dat' WITH (DELIMITER='|', HEADER='false');"
    echo "$QUERY"
    echo "$QUERY" | $HEAVYSQL
  done
fi

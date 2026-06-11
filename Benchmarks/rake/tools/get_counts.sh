#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Show row counts for each TPC-DS table.

# These utilty scripts are intended to be customized.
# Customize these values:
DATABASE=tpcds_500gb
HEAVYSQL="/opt/heavyai/bin/heavysql -p HyperInteractive $DATABASE"
TABLES="call_center catalog_page catalog_returns catalog_sales customer customer_address customer_demographics date_dim dbgen_version household_demographics income_band inventory item promotion reason ship_mode store store_returns store_sales time_dim warehouse web_page web_returns web_sales web_site"

for t in $TABLES
do
  QUERY="SELECT '$t', COUNT(*) as nrows FROM $t;"
  echo "$QUERY" | $HEAVYSQL -q
done


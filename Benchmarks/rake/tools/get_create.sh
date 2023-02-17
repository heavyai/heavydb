#!/bin/bash

# Copyright 2023 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get CREATE TABLE sql statements.
# Useful for copying optimized tables definitions.

# These utilty scripts are intended to be customized.
# Customize these values:
DATABASE=tpcds_500gb
HEAVYSQL="/opt/heavyai/bin/heavysql -p HyperInteractive $DATABASE"
TABLES="call_center catalog_page catalog_returns catalog_sales customer customer_address customer_demographics date_dim dbgen_version household_demographics income_band inventory item promotion reason ship_mode store store_returns store_sales time_dim warehouse web_page web_returns web_sales web_site"

for t in $TABLES
do
  echo "\\d $t" | $HEAVYSQL -q
done


# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM docker-internal.mapd.com/mapd/core-build-ubuntu22.04-shared-cuda12.2.2-x86_64:rc.v9.0.0

# Manually rotate repository signing keys https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
  && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
  apt-get install -y \
  libsqliteodbc \
  odbc-postgresql \
  sqlite3 \
  gdb \
  vim \
  unixodbc \
  libsasl2-2 \
  libsasl2-modules-gssapi-mit \
  tcl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://s3.amazonaws.com/redshift-downloads/drivers/odbc/1.4.45.1000/AmazonRedshiftODBC-64-bit-1.4.45.1000-1.x86_64.deb
RUN sudo dpkg -i ./AmazonRedshiftODBC-64-bit-1.4.45.1000-1.x86_64.deb

RUN wget https://sfc-repo.snowflakecomputing.com/odbc/linux/2.24.5/snowflake-odbc-2.24.5.x86_64.deb
RUN sudo dpkg -i ./snowflake-odbc-2.24.5.x86_64.deb

RUN wget https://storage.googleapis.com/simba-bq-release/odbc/SimbaODBCDriverforGoogleBigQuery_2.5.0.1001-Linux.tar.gz
RUN tar -xzf ./SimbaODBCDriverforGoogleBigQuery_2.5.0.1001-Linux.tar.gz && \
  sudo mkdir -p /opt/simba/ && \
  sudo tar -xzf ./SimbaODBCDriverforGoogleBigQuery_2.5.0.1001-Linux/SimbaODBCDriverforGoogleBigQuery64_2.5.0.1001.tar.gz -C  /opt/simba/ && \
  sudo mv /opt/simba/SimbaODBCDriverforGoogleBigQuery64_2.5.0.1001 /opt/simba/bigqueryodbc && \
  sudo cp ./SimbaODBCDriverforGoogleBigQuery_2.5.0.1001-Linux/GoogleBigQueryODBC.did /opt/simba/bigqueryodbc/lib && \
  sudo cp -r /opt/simba/bigqueryodbc/ErrorMessages/en-US /opt/simba/bigqueryodbc/lib/
ENV bigquery_project="mapd-engineering"

RUN wget https://downloads.cloudera.com/connectors/ClouderaHive_ODBC-2.7.0.1002/Debian/clouderahiveodbc_2.7.0.1002-2_amd64.deb
RUN sudo dpkg -i ./clouderahiveodbc_2.7.0.1002-2_amd64.deb

RUN mkdir -p /usr/lib/odbc-drivers


# TODO: Uncomment below COPY and update setup documentation with instructions
# on how to download OmniSci ODBC driver when the ODBC driver issue is fixed
# COPY ./libOmniSciODBC.so /usr/lib/odbc-drivers/

COPY ./odbc.ini /etc/odbc.ini
COPY ./odbcinst.ini /etc/odbcinst.ini

# Build sqlite driver from source
COPY ./build_sqlite_odbc_driver.sh /build_sqlite_odbc_driver.sh
RUN /bin/bash /build_sqlite_odbc_driver.sh

RUN mkdir -p /var/lib/odbc-fsi-dev/

RUN sqlite3 /var/lib/odbc-fsi-dev/test_db.sqlite

RUN echo "source /usr/local/mapd-deps/mapd-deps.sh" >> /root/.bashrc

# Fix redshift missing symbol error:
# [Amazon][ODBC] (11560) Unable to locate SQLGetPrivateProfileString function
RUN echo "" >> /opt/amazon/redshiftodbc/lib/64/amazon.redshiftodbc.ini
RUN echo "ODBCInstLib=/usr/local/mapd-deps/lib/libiodbcinst.so.2" >> /opt/amazon/redshiftodbc/lib/64/amazon.redshiftodbc.ini

# Fixup hive ODBC driver
RUN sed -ie 's/ODBCInstLib=libiodbcinst.so/#ODBCInstLib=libiodbcinst.so/g' /opt/cloudera/hiveodbc/lib/64/cloudera.hiveodbc.ini
RUN echo "ODBCInstLib=/usr/local/mapd-deps/lib/libiodbcinst.so.2" >> /opt/cloudera/hiveodbc/lib/64/cloudera.hiveodbc.ini

WORKDIR /odbc-fsi-src

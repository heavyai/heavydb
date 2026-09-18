# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE_TAG=master

FROM docker-internal.mapd.com/mapd/mapd-render:$BASE_IMAGE_TAG

RUN apt-get update && \
    apt-get install -y \
    odbc-postgresql \
    unixodbc \
    libsasl2-2 \
    libsasl2-modules-gssapi-mit && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://s3.amazonaws.com/redshift-downloads/drivers/odbc/1.4.65.1000/AmazonRedshiftODBC-64-bit-1.4.65.1000-1.x86_64.deb && \
    dpkg -i ./AmazonRedshiftODBC-64-bit-1.4.65.1000-1.x86_64.deb && \
    rm ./AmazonRedshiftODBC-64-bit-1.4.65.1000-1.x86_64.deb

RUN wget https://sfc-repo.snowflakecomputing.com/odbc/linux/2.25.12/snowflake-odbc-2.25.12.x86_64.deb && \
    dpkg -i ./snowflake-odbc-2.25.12.x86_64.deb && \
    rm ./snowflake-odbc-2.25.12.x86_64.deb

RUN wget https://storage.googleapis.com/simba-bq-release/odbc/SimbaODBCDriverforGoogleBigQuery_3.0.0.1001-Linux.tar.gz && \
    tar -xzf ./SimbaODBCDriverforGoogleBigQuery_3.0.0.1001-Linux.tar.gz && \
    mkdir -p /opt/simba/ && \
    tar -xzf ./SimbaODBCDriverforGoogleBigQuery_3.0.0.1001-Linux/SimbaODBCDriverforGoogleBigQuery64_3.0.0.1001.tar.gz -C  /opt/simba/ && \
    mv /opt/simba/SimbaODBCDriverforGoogleBigQuery64_3.0.0.1001 /opt/simba/bigqueryodbc && \
    cp ./SimbaODBCDriverforGoogleBigQuery_3.0.0.1001-Linux/GoogleBigQueryODBC.did /opt/simba/bigqueryodbc/lib && \
    cp -r /opt/simba/bigqueryodbc/ErrorMessages/en-US /opt/simba/bigqueryodbc/lib/ && \
    rm -rf ./SimbaODBCDriverforGoogleBigQuery_3.0.0.1001*

RUN wget https://downloads.cloudera.com/connectors/hive-2.5.25.1020/Debian/clouderahiveodbc_2.5.25.1020-2_amd64.deb && \
    dpkg -i ./clouderahiveodbc_2.5.25.1020-2_amd64.deb && \
    rm ./clouderahiveodbc_2.5.25.1020-2_amd64.deb

COPY ./odbcinst.ini /etc/odbcinst.ini

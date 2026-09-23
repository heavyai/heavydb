# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:22.04

# Prevent apt-get asking for input
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    ssh \
    pdsh \
    openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

ARG hadoop_version=3.3.5
ARG hive_version=4.0.1
RUN wget -q https://dlcdn.apache.org/hadoop/common/hadoop-${hadoop_version}/hadoop-${hadoop_version}.tar.gz
RUN wget -q https://dlcdn.apache.org/hive/hive-${hive_version}/apache-hive-${hive_version}-bin.tar.gz
RUN tar -xzf /hadoop-${hadoop_version}.tar.gz
RUN tar -xzf /apache-hive-${hive_version}-bin.tar.gz

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV HADOOP_HOME=hadoop-${hadoop_version}
ENV HIVE_HOME=/apache-hive-${hive_version}-bin
ENV PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$HIVE_HOME/bin:$PATH"
ENV JVM_ARGS="-Xms1024m -Xmx1024m"

# Configuration settings for hadoop/hive
RUN touch \
    ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh \
    ${HIVE_HOME}/conf/hive-env.sh
RUN echo "export JAVA_HOME=${JAVA_HOME}" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
RUN echo "export HADOOP_CLIENT_OPTS=\"-Xmx1024m ${HADOOP_CLIENT_OPTS}\"" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
RUN echo "export HADOOP_HOME=$HADOOP_HOME" >> ${HIVE_HOME}/conf/hive-env.sh
COPY ./hive-site.xml ${HIVE_HOME}/conf/
COPY ./mapred-site.xml ${HADOOP_HOME}/etc/hadoop/

# Ensure consistent guava.jar & setup hive warehouse
COPY ./hive-setup.sh /hive-setup.sh
RUN . /hive-setup.sh

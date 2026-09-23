# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Hive and Hadoop must use the most recent and same version of guava-X.Y.Z.jar
hive_lib_path=$HIVE_HOME/lib
hadoop_lib_path=$HADOOP_HOME/share/hadoop/hdfs/lib
old_guava=$(find $hive_lib_path $hadoop_lib_path -name "guava*.jar" -printf '%f %p\n' | sort -V -k1,1 | cut -f 2- -d ' ' | head -n 1)
new_guava=$(find $hive_lib_path $hadoop_lib_path -name "guava*.jar" -printf '%f %p\n' | sort -V -k1,1 | cut -f 2- -d ' ' | tail -n 1)
mv $old_guava ${old_guava}.backup
cp $new_guava $(dirname $old_guava)

# Add hive/hadoop/java binaries to path and intialize hive warehouse/metastore
$HADOOP_HOME/bin/hdfs dfs -mkdir /HiveWarehouse
$HADOOP_HOME/bin/hdfs dfs -chmod g+w /tmp/ /HiveWarehouse/
$HIVE_HOME/bin/schematool -dbType derby -initSchema
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is a modified derivative of Apache Calcite's org.apache.calcite.sql.ddl.SqlDropTable.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlDrop;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

import java.util.List;

/**
 * Class that encapsulates all information associated with a DROP TABLE DDL command.
 */
public class SqlDropTable extends SqlDrop implements JsonSerializableDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("DROP_TABLE", SqlKind.DROP_TABLE);

  @Expose
  private boolean ifExists;
  @Expose
  private String tableName;
  @Expose
  private String command;

  public SqlDropTable(
          final SqlParserPos pos, final boolean ifExists, final String tableName) {
    super(OPERATOR, pos, ifExists);
    this.ifExists = ifExists;
    this.tableName = tableName;
    this.command = OPERATOR.getName();
  }

  @Override
  public List<SqlNode> getOperandList() {
    return null;
  }

  @Override
  public String toString() {
    return toJsonString();
  }
}

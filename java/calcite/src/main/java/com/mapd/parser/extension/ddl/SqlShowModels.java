/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.extension.ddl;

import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.parser.SqlParserPos;

/**
 * Class that encapsulates all information associated with a SHOW MODELS DDL command.
 */
public class SqlShowModels extends SqlCustomDdl {
  private static final SqlOperator OPERATOR =
          new SqlSpecialOperator("SHOW_MODELS", SqlKind.OTHER_DDL);

  public SqlShowModels(final SqlParserPos pos) {
    super(OPERATOR, pos);
  }
}

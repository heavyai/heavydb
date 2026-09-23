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
 * This file is a modified derivative of Apache Calcite's org.apache.calcite.sql.fun.SqlSingleValueAggFunction.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.calcite.sql.fun;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlSplittableAggFunction;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.util.Optionality;

import com.google.common.collect.ImmutableList;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.List;

/**
 * <code>SINGLE_VALUE</code> aggregate function returns the input value if there
 * is only one value in the input; Otherwise it triggers a run-time error.
 */
public class SqlSingleValueAggFunction extends SqlAggFunction {
  //~ Instance fields --------------------------------------------------------

  @Deprecated // to be removed before 2.0
  private final RelDataType type;

  //~ Constructors -----------------------------------------------------------

  public SqlSingleValueAggFunction(
      RelDataType type) {
    super(
        "SINGLE_VALUE",
        null,
        SqlKind.SINGLE_VALUE,
        ReturnTypes.ARG0_NULLABLE_IF_EMPTY,
        null,
        OperandTypes.ANY,
        SqlFunctionCategory.SYSTEM,
        false,
        false,
        Optionality.FORBIDDEN);
    this.type = type;
  }

  //~ Methods ----------------------------------------------------------------

  @Override public boolean allowsFilter() {
    return false;
  }

  @SuppressWarnings("deprecation")
  @Override public List<RelDataType> getParameterTypes(RelDataTypeFactory typeFactory) {
    return ImmutableList.of(type);
  }

  @SuppressWarnings("deprecation")
  @Override public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
    return type;
  }

  @Deprecated // to be removed before 2.0
  public RelDataType getType() {
    return type;
  }

  @Override public <T extends Object> @Nullable T unwrap(Class<T> clazz) {
    if (clazz.isInstance(SqlSplittableAggFunction.SelfSplitter.INSTANCE)) {
      return clazz.cast(SqlSplittableAggFunction.SelfSplitter.INSTANCE);
    }
    return super.unwrap(clazz);
  }
  // HeavyAI: this override is intentionally disabled so that the base
  // SqlAggFunction.getDistinctOptionality() is called, which returns Optionality.OPTIONAL.
  //
  // With the override present, returning Optionality.IGNORED causes AggregateCall's
  // constructor to assert that distinct==false and throw a generic
  // IllegalArgumentException for SINGLE_VALUE(DISTINCT x).  By returning
  // Optionality.OPTIONAL instead, the AggregateCall is constructed successfully with
  // distinct=true, allowing HeavyDB's own validation to run and report a clear,
  // user-facing error.
  //
  //@Override public Optionality getDistinctOptionality() {
  //  return Optionality.IGNORED;
  //}

  @Override public SqlAggFunction getRollup() {
    return this;
  }
}

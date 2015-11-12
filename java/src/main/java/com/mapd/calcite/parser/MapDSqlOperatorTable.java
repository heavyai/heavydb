/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.calcite.parser;

import com.google.common.collect.ImmutableList;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.util.ListSqlOperatorTable;

/**
 *
 * @author michael
 */
public class MapDSqlOperatorTable extends ChainedSqlOperatorTable {

  /**
   * Mock operator table for testing purposes. Contains the standard SQL operator table, plus a list
   * of operators.
   */
  //~ Instance fields --------------------------------------------------------
  private final ListSqlOperatorTable listOpTab;

  //~ Constructors -----------------------------------------------------------
  public MapDSqlOperatorTable(SqlOperatorTable parentTable) {
    super(ImmutableList.of(parentTable, new ListSqlOperatorTable()));
    listOpTab = (ListSqlOperatorTable) tableList.get(1);
  }

  //~ Methods ----------------------------------------------------------------
  /**
   * Adds an operator to this table.
   */
  public void addOperator(SqlOperator op) {
    listOpTab.add(op);
  }

  // MAT Nov 11 2015
  // These are example of how to add custom functions
  // left in as a starting point for when we need them
  public static void addRamp(MapDSqlOperatorTable opTab) {
    // Don't use anonymous inner classes. They can't be instantiated
    // using reflection when we are deserializing from JSON.
    opTab.addOperator(new RampFunction());
    opTab.addOperator(new DedupFunction());
  }

  /**
   * "RAMP" user-defined function.
   */
  public static class RampFunction extends SqlFunction {

    public RampFunction() {
      super("RAMP",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.NUMERIC,
              SqlFunctionCategory.USER_DEFINED_FUNCTION);
    }

    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory
              = opBinding.getTypeFactory();
      return typeFactory.builder()
              .add("I", SqlTypeName.INTEGER)
              .build();
    }
  }

  /**
   * "DEDUP" user-defined function.
   */
  public static class DedupFunction extends SqlFunction {

    public DedupFunction() {
      super("DEDUP",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.VARIADIC,
              SqlFunctionCategory.USER_DEFINED_FUNCTION);
    }

    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory
              = opBinding.getTypeFactory();
      return typeFactory.builder()
              .add("NAME", SqlTypeName.VARCHAR, 1024)
              .build();
    }
  }
}

// End MapDSqlOperatorTable.java

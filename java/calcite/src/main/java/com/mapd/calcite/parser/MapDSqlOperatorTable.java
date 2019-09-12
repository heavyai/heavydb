/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.calcite.parser;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.mapd.parser.server.ExtensionFunction;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlArrayValueConstructor;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.util.ListSqlOperatorTable;
import org.apache.calcite.sql.util.ReflectiveSqlOperatorTable;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

class CaseInsensitiveListSqlOperatorTable extends ListSqlOperatorTable {
  @Override
  public void lookupOperatorOverloads(SqlIdentifier opName,
          SqlFunctionCategory category,
          SqlSyntax syntax,
          List<SqlOperator> operatorList,
          SqlNameMatcher nameMatcher) {
    for (SqlOperator operator : this.getOperatorList()) {
      if (operator.getSyntax() != syntax) {
        continue;
      }
      if (!opName.isSimple()
              || !nameMatcher.matches(operator.getName(), opName.getSimple())) {
        continue;
      }
      SqlFunctionCategory functionCategory;
      if (operator instanceof SqlFunction) {
        functionCategory = ((SqlFunction) operator).getFunctionType();
      } else {
        functionCategory = SqlFunctionCategory.SYSTEM;
      }
      if (category != functionCategory
              && category != SqlFunctionCategory.USER_DEFINED_FUNCTION) {
        continue;
      }
      operatorList.add(operator);
    }
  }
}

/**
 *
 * @author michael
 */
public class MapDSqlOperatorTable extends ChainedSqlOperatorTable {
  public static final SqlArrayValueConstructorAllowingEmpty ARRAY_VALUE_CONSTRUCTOR =
          new SqlArrayValueConstructorAllowingEmpty();

  static {
    try {
      // some nasty bit to remove the std APPROX_COUNT_DISTINCT function definition
      {
        Field f = ReflectiveSqlOperatorTable.class.getDeclaredField(
                "caseSensitiveOperators");
        f.setAccessible(true);
        Multimap operators = (Multimap) f.get(SqlStdOperatorTable.instance());
        for (Iterator i = operators.entries().iterator(); i.hasNext();) {
          Map.Entry entry = (Map.Entry) i.next();
          if (entry.getValue() == SqlStdOperatorTable.APPROX_COUNT_DISTINCT
                  || entry.getValue() == SqlStdOperatorTable.AVG
                  || entry.getValue() == SqlStdOperatorTable.ARRAY_VALUE_CONSTRUCTOR) {
            i.remove();
          }
        }
      }

      {
        Field f = ReflectiveSqlOperatorTable.class.getDeclaredField(
                "caseInsensitiveOperators");
        f.setAccessible(true);
        Multimap operators = (Multimap) f.get(SqlStdOperatorTable.instance());
        for (Iterator i = operators.entries().iterator(); i.hasNext();) {
          Map.Entry entry = (Map.Entry) i.next();
          if (entry.getValue() == SqlStdOperatorTable.APPROX_COUNT_DISTINCT
                  || entry.getValue() == SqlStdOperatorTable.AVG
                  || entry.getValue() == SqlStdOperatorTable.ARRAY_VALUE_CONSTRUCTOR) {
            i.remove();
          }
        }
      }

      SqlStdOperatorTable.instance().register(ARRAY_VALUE_CONSTRUCTOR);

    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // register our approx count distinct against std table
    // SqlStdOperatorTable.instance().register(new ApproxCountDistinct());
  }

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDSqlOperatorTable.class);

  /**
   * Mock operator table for testing purposes. Contains the standard SQL operator
   * table, plus a list of operators.
   */
  // ~ Instance fields --------------------------------------------------------
  private final ListSqlOperatorTable listOpTab;

  // ~ Constructors -----------------------------------------------------------
  public MapDSqlOperatorTable(SqlOperatorTable parentTable) {
    super(ImmutableList.of(parentTable, new CaseInsensitiveListSqlOperatorTable()));
    listOpTab = (ListSqlOperatorTable) tableList.get(1);
  }

  // ~ Methods ----------------------------------------------------------------
  /**
   * Adds an operator to this table.
   *
   * @param op
   */
  public void addOperator(SqlOperator op) {
    listOpTab.add(op);
  }

  public static void addUDF(
          MapDSqlOperatorTable opTab, final Map<String, ExtensionFunction> extSigs) {
    // Don't use anonymous inner classes. They can't be instantiated
    // using reflection when we are deserializing from JSON.
    // opTab.addOperator(new RampFunction());
    // opTab.addOperator(new DedupFunction());
    opTab.addOperator(new RowCopier()); // Table UDF prototype
    opTab.addOperator(new MyUDFFunction());
    opTab.addOperator(new PgUnnest());
    opTab.addOperator(new Any());
    opTab.addOperator(new All());
    opTab.addOperator(new Now());
    opTab.addOperator(new Datetime());
    opTab.addOperator(new PgExtract());
    opTab.addOperator(new Dateadd());
    opTab.addOperator(new Datediff());
    opTab.addOperator(new Datepart());
    opTab.addOperator(new PgDateTrunc());
    opTab.addOperator(new Length());
    opTab.addOperator(new CharLength());
    opTab.addOperator(new KeyForString());
    opTab.addOperator(new ArrayLength());
    opTab.addOperator(new PgILike());
    opTab.addOperator(new RegexpLike());
    opTab.addOperator(new Likely());
    opTab.addOperator(new Unlikely());
    opTab.addOperator(new Sign());
    opTab.addOperator(new Truncate());
    opTab.addOperator(new ST_Contains());
    opTab.addOperator(new ST_Intersects());
    opTab.addOperator(new ST_Disjoint());
    opTab.addOperator(new ST_Within());
    opTab.addOperator(new ST_DWithin());
    opTab.addOperator(new ST_DFullyWithin());
    opTab.addOperator(new ST_Distance());
    opTab.addOperator(new ST_MaxDistance());
    opTab.addOperator(new ST_GeogFromText());
    opTab.addOperator(new ST_GeomFromText());
    opTab.addOperator(new ST_Transform());
    opTab.addOperator(new ST_X());
    opTab.addOperator(new ST_Y());
    opTab.addOperator(new ST_XMin());
    opTab.addOperator(new ST_XMax());
    opTab.addOperator(new ST_YMin());
    opTab.addOperator(new ST_YMax());
    opTab.addOperator(new ST_PointN());
    opTab.addOperator(new ST_StartPoint());
    opTab.addOperator(new ST_EndPoint());
    opTab.addOperator(new ST_Length());
    opTab.addOperator(new ST_Perimeter());
    opTab.addOperator(new ST_Area());
    opTab.addOperator(new ST_NPoints());
    opTab.addOperator(new ST_NRings());
    opTab.addOperator(new ST_SRID());
    opTab.addOperator(new ST_SetSRID());
    opTab.addOperator(new ST_Point());
    opTab.addOperator(new CastToGeography());
    opTab.addOperator(new OffsetInFragment());
    opTab.addOperator(new ApproxCountDistinct());
    opTab.addOperator(new MapDAvg());
    opTab.addOperator(new Sample());
    opTab.addOperator(new LastSample());
    // MapD_Geo* are deprecated in place of the OmniSci_Geo_ varietals
    opTab.addOperator(new MapD_GeoPolyBoundsPtr());
    opTab.addOperator(new MapD_GeoPolyRenderGroup());
    opTab.addOperator(new OmniSci_Geo_PolyBoundsPtr());
    opTab.addOperator(new OmniSci_Geo_PolyRenderGroup());
    opTab.addOperator(new convert_meters_to_pixel_width());
    opTab.addOperator(new convert_meters_to_pixel_height());
    opTab.addOperator(new is_point_in_view());
    opTab.addOperator(new is_point_size_in_view());
    opTab.addOperator(new usTimestamp());
    opTab.addOperator(new nsTimestamp());
    if (extSigs == null) {
      return;
    }
    HashSet<String> demangledNames = new HashSet<String>();
    for (Map.Entry<String, ExtensionFunction> extSig : extSigs.entrySet()) {
      final String demangledName = dropSuffix(extSig.getKey());
      if (demangledNames.contains(demangledName)) {
        continue;
      }
      demangledNames.add(demangledName);
      opTab.addOperator(new ExtFunction(extSig.getKey(), extSig.getValue()));
    }
  }

  private static String dropSuffix(final String str) {
    int suffix_idx = str.indexOf("__");
    if (suffix_idx == -1) {
      return str;
    }
    assert suffix_idx > 0;
    return str.substring(0, suffix_idx - 1);
  }

  public static class SqlArrayValueConstructorAllowingEmpty
          extends SqlArrayValueConstructor {
    @Override
    protected RelDataType getComponentType(
            RelDataTypeFactory typeFactory, List<RelDataType> argTypes) {
      if (argTypes.isEmpty()) {
        return typeFactory.createSqlType(SqlTypeName.NULL);
      }
      return super.getComponentType(typeFactory, argTypes);
    }

    @Override
    public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
      if (callBinding.operands().isEmpty()) {
        return true;
      }
      return super.checkOperandTypes(callBinding, throwOnFailure);
    }
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

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.builder().add("I", SqlTypeName.INTEGER).build();
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

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.builder().add("NAME", SqlTypeName.VARCHAR, 1024).build();
    }
  }

  /**
   * "MyUDFFunction" user-defined function test. our udf's will look like system
   * functions to calcite as it has no access to the code
   */
  public static class MyUDFFunction extends SqlFunction {
    public MyUDFFunction() {
      super("MyUDF",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING_STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BIGINT);
    }
  }

  /**
   * Table-level functions
   */
  public static class RowCopier extends SqlFunction {
    public RowCopier() {
      super("ROW_COPIER",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.USER_DEFINED_TABLE_FUNCTION);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      return opBinding.getCursorOperand(0);
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> sig_family = new java.util.ArrayList<SqlTypeFamily>();
      sig_family.add(SqlTypeFamily.CURSOR);
      sig_family.add(SqlTypeFamily.ANY);
      return sig_family;
    }
  }

  /* Postgres-style UNNEST */
  public static class PgUnnest extends SqlFunction {
    public PgUnnest() {
      super("PG_UNNEST",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.ARRAY,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      RelDataType elem_type = opBinding.getOperandType(0).getComponentType();
      assert elem_type != null;
      return elem_type;
    }
  }

  /* ANY qualifier */
  public static class Any extends SqlFunction {
    public Any() {
      super("PG_ANY",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.ARRAY,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      RelDataType elem_type = opBinding.getOperandType(0).getComponentType();
      assert elem_type != null;
      return elem_type;
    }
  }

  /* ALL qualifier */
  public static class All extends SqlFunction {
    public All() {
      super("PG_ALL",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.ARRAY,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      RelDataType elem_type = opBinding.getOperandType(0).getComponentType();
      assert elem_type != null;
      return elem_type;
    }
  }

  /* NOW() */
  public static class Now extends SqlFunction {
    public Now() {
      super("NOW",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.NILADIC,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 0;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
    }
  }

  /* DATETIME */
  public static class Datetime extends SqlFunction {
    public Datetime() {
      super("DATETIME",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(
              SqlTypeName.TIMESTAMP, opBinding.getOperandType(0).getPrecision());
    }
  }

  /* Postgres-style EXTRACT */
  public static class PgExtract extends SqlFunction {
    public PgExtract() {
      super("PG_EXTRACT",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.DATETIME),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BIGINT),
              opBinding.getOperandType(1).isNullable());
    }
  }

  public static class Datepart extends SqlFunction {
    public Datepart() {
      super("DATEPART",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.DATETIME),
              SqlFunctionCategory.TIMEDATE);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BIGINT),
              opBinding.getOperandType(1).isNullable());
    }
  }

  public static class Dateadd extends SqlFunction {
    public Dateadd() {
      super("DATEADD",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.STRING,
                      SqlTypeFamily.INTEGER,
                      SqlTypeFamily.DATETIME),
              SqlFunctionCategory.TIMEDATE);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(
                      SqlTypeName.TIMESTAMP, opBinding.getOperandType(2).getPrecision()),
              opBinding.getOperandType(2).isNullable());
    }
  }

  public static class Datediff extends SqlFunction {
    public Datediff() {
      super("DATEDIFF",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.STRING,
                      SqlTypeFamily.DATETIME,
                      SqlTypeFamily.DATETIME),
              SqlFunctionCategory.TIMEDATE);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BIGINT),
              opBinding.getOperandType(1).isNullable()
                      || opBinding.getOperandType(2).isNullable());
    }
  }

  /* Postgres-style DATE_TRUNC */
  public static class PgDateTrunc extends SqlFunction {
    public PgDateTrunc() {
      super("PG_DATE_TRUNC",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.DATETIME),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(
                      SqlTypeName.TIMESTAMP, opBinding.getOperandType(1).getPrecision()),
              opBinding.getOperandType(1).isNullable());
    }
  }

  public static class Length extends SqlFunction {
    public Length() {
      super("LENGTH",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  public static class CharLength extends SqlFunction {
    public CharLength() {
      super("CHAR_LENGTH",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  public static class KeyForString extends SqlFunction {
    public KeyForString() {
      super("KEY_FOR_STRING",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  public static class ArrayLength extends SqlFunction {
    public ArrayLength() {
      super("ARRAY_LENGTH",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.ARRAY,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  public static class PgILike extends SqlFunction {
    public PgILike() {
      super("PG_ILIKE",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(getSignatureFamilies(), new EscapeOptional()),
              SqlFunctionCategory.SYSTEM);
    }

    private static java.util.List<SqlTypeFamily> getSignatureFamilies() {
      java.util.ArrayList<SqlTypeFamily> families =
              new java.util.ArrayList<SqlTypeFamily>();
      families.add(SqlTypeFamily.STRING);
      families.add(SqlTypeFamily.STRING);
      families.add(SqlTypeFamily.STRING);
      return families;
    }

    private static class EscapeOptional
            implements java.util.function.Predicate<Integer>, Predicate<Integer> {
      @Override
      public boolean test(Integer t) {
        return apply(t);
      }

      @Override
      public boolean apply(Integer t) {
        return t == 2;
      }
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    }
  }

  public static class RegexpLike extends SqlFunction {
    public RegexpLike() {
      super("REGEXP_LIKE",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(getSignatureFamilies(), new EscapeOptional()),
              SqlFunctionCategory.SYSTEM);
    }

    private static java.util.List<SqlTypeFamily> getSignatureFamilies() {
      java.util.ArrayList<SqlTypeFamily> families =
              new java.util.ArrayList<SqlTypeFamily>();
      families.add(SqlTypeFamily.STRING);
      families.add(SqlTypeFamily.STRING);
      families.add(SqlTypeFamily.STRING);
      return families;
    }

    private static class EscapeOptional
            implements java.util.function.Predicate<Integer>, Predicate<Integer> {
      @Override
      public boolean test(Integer t) {
        return apply(t);
      }

      public boolean apply(Integer t) {
        return t == 2;
      }
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    }
  }

  public static class Likely extends SqlFunction {
    public Likely() {
      super("LIKELY",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.BOOLEAN,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      return opBinding.getOperandType(0);
    }
  }

  public static class Unlikely extends SqlFunction {
    public Unlikely() {
      super("UNLIKELY",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.BOOLEAN,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      return opBinding.getOperandType(0);
    }
  }

  public static class Sign extends SqlFunction {
    public Sign() {
      super("SIGN",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.NUMERIC,
              SqlFunctionCategory.NUMERIC);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      return opBinding.getOperandType(0);
    }
  }

  static class Truncate extends SqlFunction {
    Truncate() {
      super("TRUNCATE",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.NUMERIC);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      return opBinding.getOperandType(0);
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> truncate_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      truncate_sig.add(SqlTypeFamily.NUMERIC);
      truncate_sig.add(SqlTypeFamily.INTEGER);
      return truncate_sig;
    }
  }

  static class ST_Contains extends SqlFunction {
    ST_Contains() {
      super("ST_Contains",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BOOLEAN),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_contains_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_contains_sig.add(SqlTypeFamily.ANY);
      st_contains_sig.add(SqlTypeFamily.ANY);
      return st_contains_sig;
    }
  }

  static class ST_Intersects extends SqlFunction {
    ST_Intersects() {
      super("ST_Intersects",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BOOLEAN),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_intersects_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_intersects_sig.add(SqlTypeFamily.ANY);
      st_intersects_sig.add(SqlTypeFamily.ANY);
      return st_intersects_sig;
    }
  }

  static class ST_Disjoint extends SqlFunction {
    ST_Disjoint() {
      super("ST_Disjoint",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BOOLEAN),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_disjoint_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_disjoint_sig.add(SqlTypeFamily.ANY);
      st_disjoint_sig.add(SqlTypeFamily.ANY);
      return st_disjoint_sig;
    }
  }

  static class ST_Within extends SqlFunction {
    ST_Within() {
      super("ST_Within",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BOOLEAN),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_within_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_within_sig.add(SqlTypeFamily.ANY);
      st_within_sig.add(SqlTypeFamily.ANY);
      return st_within_sig;
    }
  }

  static class ST_DWithin extends SqlFunction {
    ST_DWithin() {
      super("ST_DWithin",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 3;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BOOLEAN),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable()
                      || opBinding.getOperandType(2).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_dwithin_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_dwithin_sig.add(SqlTypeFamily.ANY);
      st_dwithin_sig.add(SqlTypeFamily.ANY);
      st_dwithin_sig.add(SqlTypeFamily.NUMERIC);
      return st_dwithin_sig;
    }
  }

  static class ST_DFullyWithin extends SqlFunction {
    ST_DFullyWithin() {
      super("ST_DFullyWithin",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 3;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.BOOLEAN),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable()
                      || opBinding.getOperandType(2).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_dwithin_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_dwithin_sig.add(SqlTypeFamily.ANY);
      st_dwithin_sig.add(SqlTypeFamily.ANY);
      st_dwithin_sig.add(SqlTypeFamily.NUMERIC);
      return st_dwithin_sig;
    }
  }

  static class ST_Distance extends SqlFunction {
    ST_Distance() {
      super("ST_Distance",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_distance_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_distance_sig.add(SqlTypeFamily.ANY);
      st_distance_sig.add(SqlTypeFamily.ANY);
      return st_distance_sig;
    }
  }

  static class ST_MaxDistance extends SqlFunction {
    ST_MaxDistance() {
      super("ST_MaxDistance",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(signature()),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable()
                      || opBinding.getOperandType(1).isNullable());
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_maxdistance_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_maxdistance_sig.add(SqlTypeFamily.ANY);
      st_maxdistance_sig.add(SqlTypeFamily.ANY);
      return st_maxdistance_sig;
    }
  }

  static class ST_GeogFromText extends SqlFunction {
    ST_GeogFromText() {
      super("ST_GeogFromText",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.or(OperandTypes.family(SqlTypeFamily.ANY),
                      OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER)),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  static class ST_GeomFromText extends SqlFunction {
    ST_GeomFromText() {
      super("ST_GeomFromText",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.or(OperandTypes.family(SqlTypeFamily.ANY),
                      OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER)),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  static class ST_Transform extends SqlFunction {
    ST_Transform() {
      super("ST_Transform",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_X extends SqlFunction {
    ST_X() {
      super("ST_X",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_Y extends SqlFunction {
    ST_Y() {
      super("ST_Y",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_XMin extends SqlFunction {
    ST_XMin() {
      super("ST_XMin",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_XMax extends SqlFunction {
    ST_XMax() {
      super("ST_XMax",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_YMin extends SqlFunction {
    ST_YMin() {
      super("ST_YMin",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_YMax extends SqlFunction {
    ST_YMax() {
      super("ST_YMax",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_PointN extends SqlFunction {
    ST_PointN() {
      super("ST_PointN",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_EndPoint extends SqlFunction {
    ST_EndPoint() {
      super("ST_EndPoint",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_StartPoint extends SqlFunction {
    ST_StartPoint() {
      super("ST_StartPoint",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_Length extends SqlFunction {
    ST_Length() {
      super("ST_Length",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_Perimeter extends SqlFunction {
    ST_Perimeter() {
      super("ST_Perimeter",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_Area extends SqlFunction {
    ST_Area() {
      super("ST_Area",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.DOUBLE),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_NPoints extends SqlFunction {
    ST_NPoints() {
      super("ST_NPoints",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_NRings extends SqlFunction {
    ST_NRings() {
      super("ST_NRings",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_SRID extends SqlFunction {
    ST_SRID() {
      super("ST_SRID",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_SetSRID extends SqlFunction {
    ST_SetSRID() {
      super("ST_SetSRID",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(
              typeFactory.createSqlType(SqlTypeName.INTEGER),
              opBinding.getOperandType(0).isNullable());
    }
  }

  static class ST_Point extends SqlFunction {
    ST_Point() {
      super("ST_Point",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 2;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  static class CastToGeography extends SqlFunction {
    CastToGeography() {
      super("CastToGeography",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  /* OFFSET_IN_FRAGMENT() */
  public static class OffsetInFragment extends SqlFunction {
    public OffsetInFragment() {
      super("OFFSET_IN_FRAGMENT",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.NILADIC,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 0;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  static class ApproxCountDistinct extends SqlAggFunction {
    ApproxCountDistinct() {
      super("APPROX_COUNT_DISTINCT",
              null,
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.or(OperandTypes.family(SqlTypeFamily.ANY),
                      OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER)),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BIGINT);
    }
  }

  static class MapDAvg extends SqlAggFunction {
    MapDAvg() {
      super("AVG",
              null,
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.NUMERIC),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
    }
  }

  public static class Sample extends SqlAggFunction {
    public Sample() {
      super("SAMPLE",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.ANY,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      return opBinding.getOperandType(0);
    }
  }

  // for backwards compatibility
  public static class LastSample extends SqlAggFunction {
    public LastSample() {
      super("LAST_SAMPLE",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.ANY,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      return opBinding.getOperandType(0);
    }
  }

  static class ExtFunction extends SqlFunction {
    ExtFunction(final String name, final ExtensionFunction sig) {
      super(name,
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(toSqlSignature(sig)),
              sig.isRowUdf() ? SqlFunctionCategory.SYSTEM
                             : SqlFunctionCategory.USER_DEFINED_TABLE_FUNCTION);
      isRowUdf = sig.isRowUdf();
      if (isRowUdf) {
        ret = toSqlTypeName(sig.getRet());
      } else {
        ret = null;
      }
    }

    private static java.util.List<SqlTypeFamily> toSqlSignature(
            final ExtensionFunction sig) {
      java.util.List<SqlTypeFamily> sql_sig = new java.util.ArrayList<SqlTypeFamily>();
      boolean isRowUdf = sig.isRowUdf();
      for (int arg_idx = 0; arg_idx < sig.getArgs().size(); ++arg_idx) {
        final ExtensionFunction.ExtArgumentType arg_type = sig.getArgs().get(arg_idx);
        if (isRowUdf) {
          sql_sig.add(toSqlTypeName(arg_type).getFamily());
          if (isPointerType(arg_type)) {
            ++arg_idx;
          }
        } else {
          if (isPointerType(arg_type)) {
            /* TODO: eliminate using getValueType */
            sql_sig.add(toSqlTypeName(getValueType(arg_type)).getFamily());
          } else {
            sql_sig.add(toSqlTypeName(arg_type).getFamily());
          }
        }
      }
      return sql_sig;
    }

    private static boolean isPointerType(final ExtensionFunction.ExtArgumentType type) {
      return type == ExtensionFunction.ExtArgumentType.PInt8
              || type == ExtensionFunction.ExtArgumentType.PInt16
              || type == ExtensionFunction.ExtArgumentType.PInt32
              || type == ExtensionFunction.ExtArgumentType.PInt64
              || type == ExtensionFunction.ExtArgumentType.PFloat
              || type == ExtensionFunction.ExtArgumentType.PDouble;
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      if (isRowUdf) {
        final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        return typeFactory.createTypeWithNullability(
                typeFactory.createSqlType(ret), true);
      } else {
        assert opBinding.getOperandCount() == 2;
        return opBinding.getCursorOperand(0);
      }
    }

    private static ExtensionFunction.ExtArgumentType getValueType(
            final ExtensionFunction.ExtArgumentType type) {
      switch (type) {
        case PInt8:
          return ExtensionFunction.ExtArgumentType.Int8;
        case PInt16:
          return ExtensionFunction.ExtArgumentType.Int16;
        case PInt32:
          return ExtensionFunction.ExtArgumentType.Int32;
        case PInt64:
          return ExtensionFunction.ExtArgumentType.Int64;
        case PFloat:
          return ExtensionFunction.ExtArgumentType.Float;
        case PDouble:
          return ExtensionFunction.ExtArgumentType.Double;
      }
      MAPDLOGGER.error("getValueType: no value for type " + type);
      assert false;
      return null;
    }

    private static SqlTypeName toSqlTypeName(
            final ExtensionFunction.ExtArgumentType type) {
      switch (type) {
        case Bool:
          return SqlTypeName.BOOLEAN;
        case Int8:
          return SqlTypeName.TINYINT;
        case Int16:
          return SqlTypeName.SMALLINT;
        case Int32:
          return SqlTypeName.INTEGER;
        case Int64:
          return SqlTypeName.BIGINT;
        case Float:
          return SqlTypeName.FLOAT;
        case Double:
          return SqlTypeName.DOUBLE;
        case PInt8:
        case PInt16:
        case PInt32:
        case PInt64:
        case PFloat:
        case PDouble:
        case ArrayInt8:
        case ArrayInt16:
        case ArrayInt32:
        case ArrayInt64:
        case ArrayFloat:
        case ArrayDouble:
          return SqlTypeName.ARRAY;
        case GeoPoint:
        case GeoLineString:
        case GeoPolygon:
        case GeoMultiPolygon:
          return SqlTypeName.GEOMETRY;
        case Cursor:
          return SqlTypeName.CURSOR;
      }
      MAPDLOGGER.error("toSqlTypeName: unknown type " + type);
      assert false;
      return null;
    }
    private final boolean isRowUdf;
    private final SqlTypeName ret;
  }

  //
  // Internal accessors for in-situ poly render queries
  //
  // The MapD_* varietals are deprecated. The OmniSci_Geo_* ones should be used
  // instead
  //

  static class MapD_GeoPolyBoundsPtr extends SqlFunction {
    MapD_GeoPolyBoundsPtr() {
      super("MapD_GeoPolyBoundsPtr",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BIGINT);
    }
  }

  static class MapD_GeoPolyRenderGroup extends SqlFunction {
    MapD_GeoPolyRenderGroup() {
      super("MapD_GeoPolyRenderGroup",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  static class OmniSci_Geo_PolyBoundsPtr extends SqlFunction {
    OmniSci_Geo_PolyBoundsPtr() {
      super("OmniSci_Geo_PolyBoundsPtr",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BIGINT);
    }
  }

  static class OmniSci_Geo_PolyRenderGroup extends SqlFunction {
    OmniSci_Geo_PolyRenderGroup() {
      super("OmniSci_Geo_PolyRenderGroup",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
    }
  }

  static class convert_meters_to_pixel_width extends SqlFunction {
    convert_meters_to_pixel_width() {
      super("convert_meters_to_pixel_width",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.ANY,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 6;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
    }
  }

  static class convert_meters_to_pixel_height extends SqlFunction {
    convert_meters_to_pixel_height() {
      super("convert_meters_to_pixel_height",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.ANY,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 6;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
    }
  }

  static class is_point_in_view extends SqlFunction {
    is_point_in_view() {
      super("is_point_in_view",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 5;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    }
  }

  static class is_point_size_in_view extends SqlFunction {
    is_point_size_in_view() {
      super("is_point_size_in_view",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.family(SqlTypeFamily.ANY,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC,
                      SqlTypeFamily.NUMERIC),
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 6;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    }
  }

  public static class usTimestamp extends SqlFunction {
    public usTimestamp() {
      super("usTIMESTAMP",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.TIMESTAMP, 6);
    }
  }

  public static class nsTimestamp extends SqlFunction {
    public nsTimestamp() {
      super("nsTIMESTAMP",
              SqlKind.OTHER_FUNCTION,
              null,
              null,
              OperandTypes.STRING,
              SqlFunctionCategory.SYSTEM);
    }

    @Override
    public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
      assert opBinding.getOperandCount() == 1;
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createSqlType(SqlTypeName.TIMESTAMP, 9);
    }
  }
}

// End MapDSqlOperatorTable.java

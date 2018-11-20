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

import java.lang.reflect.Field;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.mapd.parser.server.ExtensionFunction;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.util.ListSqlOperatorTable;
import org.apache.calcite.sql.util.ReflectiveSqlOperatorTable;

class CaseInsensitiveListSqlOperatorTable extends ListSqlOperatorTable {
  @Override
  public void lookupOperatorOverloads(SqlIdentifier opName,
          SqlFunctionCategory category,
          SqlSyntax syntax,
          List<SqlOperator> operatorList) {
    for (SqlOperator operator : this.getOperatorList()) {
      if (operator.getSyntax() != syntax) {
        continue;
      }
      if (!opName.isSimple()
              || !operator.getName().equalsIgnoreCase(opName.getSimple())) {
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
  static {
    try {
      // some nasty bit to remove the std APPROX_COUNT_DISTINCT function definition

      Field f = ReflectiveSqlOperatorTable.class.getDeclaredField("operators");
      f.setAccessible(true);
      Multimap operators = (Multimap) f.get(SqlStdOperatorTable.instance());
      for (Iterator i = operators.entries().iterator(); i.hasNext();) {
        Map.Entry entry = (Map.Entry) i.next();
        if (entry.getValue() == SqlStdOperatorTable.APPROX_COUNT_DISTINCT) {
          i.remove();
        }
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // register our approx count distinct against std table
    SqlStdOperatorTable.instance().register(new ApproxCountDistinct());
  }

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
    opTab.addOperator(new PgILike());
    opTab.addOperator(new RegexpLike());
    opTab.addOperator(new Likely());
    opTab.addOperator(new Unlikely());
    opTab.addOperator(new Sign());
    opTab.addOperator(new Truncate());
    opTab.addOperator(new ST_Contains());
    opTab.addOperator(new ST_Intersects());
    opTab.addOperator(new ST_Within());
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
    opTab.addOperator(new CastToGeography());
    opTab.addOperator(new OffsetInFragment());
    opTab.addOperator(new ApproxCountDistinct());
    opTab.addOperator(new Sample());
    opTab.addOperator(new LastSample());
    opTab.addOperator(new MapD_GeoPolyBoundsPtr());
    opTab.addOperator(new MapD_GeoPolyRenderGroup());
    opTab.addOperator(new convert_meters_to_pixel_width());
    opTab.addOperator(new convert_meters_to_pixel_height());
    opTab.addOperator(new is_point_in_view());
    opTab.addOperator(new is_point_size_in_view());
    ////MAPD_UDF1START-DONOTCHANGETHISCOMMENT
    /* Code within this MAPD_UDF1 block is generated by mapd_udf.py script */
    opTab.addOperator(new Pyudf_d_d());
    opTab.addOperator(new Pyudf_dd_d());
    opTab.addOperator(new Pyudf_ddd_d());
    opTab.addOperator(new Pyudf_dddd_d());
    opTab.addOperator(new Pyudf_ddddd_d());
    opTab.addOperator(new Pyudf_dddddd_d());
    opTab.addOperator(new Pyudf_ddddddd_d());
    opTab.addOperator(new Pyudf_dddddddd_d());
    opTab.addOperator(new Pyudf_ddddddddd_d());
    opTab.addOperator(new Pyudf_dddddddddd_d());
    opTab.addOperator(new Pyudf_f_f());
    opTab.addOperator(new Pyudf_ff_f());
    opTab.addOperator(new Pyudf_fff_f());
    opTab.addOperator(new Pyudf_ffff_f());
    opTab.addOperator(new Pyudf_fffff_f());
    opTab.addOperator(new Pyudf_ffffff_f());
    opTab.addOperator(new Pyudf_fffffff_f());
    opTab.addOperator(new Pyudf_ffffffff_f());
    opTab.addOperator(new Pyudf_fffffffff_f());
    opTab.addOperator(new Pyudf_ffffffffff_f());
    opTab.addOperator(new Pyudf_l_l());
    opTab.addOperator(new Pyudf_ll_l());
    opTab.addOperator(new Pyudf_lll_l());
    opTab.addOperator(new Pyudf_llll_l());
    opTab.addOperator(new Pyudf_lllll_l());
    opTab.addOperator(new Pyudf_llllll_l());
    opTab.addOperator(new Pyudf_lllllll_l());
    opTab.addOperator(new Pyudf_llllllll_l());
    opTab.addOperator(new Pyudf_lllllllll_l());
    opTab.addOperator(new Pyudf_llllllllll_l());
    opTab.addOperator(new Pyudf_i_i());
    opTab.addOperator(new Pyudf_ii_i());
    opTab.addOperator(new Pyudf_iii_i());
    opTab.addOperator(new Pyudf_iiii_i());
    opTab.addOperator(new Pyudf_iiiii_i());
    opTab.addOperator(new Pyudf_iiiiii_i());
    opTab.addOperator(new Pyudf_iiiiiii_i());
    opTab.addOperator(new Pyudf_iiiiiiii_i());
    opTab.addOperator(new Pyudf_iiiiiiiii_i());
    opTab.addOperator(new Pyudf_iiiiiiiiii_i());
    opTab.addOperator(new Pyudf_d_f());
    opTab.addOperator(new Pyudf_f_d());
    opTab.addOperator(new Pyudf_dd_f());
    opTab.addOperator(new Pyudf_df_d());
    opTab.addOperator(new Pyudf_df_f());
    opTab.addOperator(new Pyudf_fd_d());
    opTab.addOperator(new Pyudf_fd_f());
    opTab.addOperator(new Pyudf_ff_d());
    opTab.addOperator(new Pyudf_d_l());
    opTab.addOperator(new Pyudf_l_d());
    opTab.addOperator(new Pyudf_dd_l());
    opTab.addOperator(new Pyudf_dl_d());
    opTab.addOperator(new Pyudf_dl_l());
    opTab.addOperator(new Pyudf_ld_d());
    opTab.addOperator(new Pyudf_ld_l());
    opTab.addOperator(new Pyudf_ll_d());
    opTab.addOperator(new Pyudf_d_i());
    opTab.addOperator(new Pyudf_i_d());
    opTab.addOperator(new Pyudf_dd_i());
    opTab.addOperator(new Pyudf_di_d());
    opTab.addOperator(new Pyudf_di_i());
    opTab.addOperator(new Pyudf_id_d());
    opTab.addOperator(new Pyudf_id_i());
    opTab.addOperator(new Pyudf_ii_d());
    opTab.addOperator(new Pyudf_l_f());
    opTab.addOperator(new Pyudf_f_l());
    opTab.addOperator(new Pyudf_ll_f());
    opTab.addOperator(new Pyudf_lf_l());
    opTab.addOperator(new Pyudf_lf_f());
    opTab.addOperator(new Pyudf_fl_l());
    opTab.addOperator(new Pyudf_fl_f());
    opTab.addOperator(new Pyudf_ff_l());
    opTab.addOperator(new Pyudf_l_i());
    opTab.addOperator(new Pyudf_i_l());
    opTab.addOperator(new Pyudf_ll_i());
    opTab.addOperator(new Pyudf_li_l());
    opTab.addOperator(new Pyudf_li_i());
    opTab.addOperator(new Pyudf_il_l());
    opTab.addOperator(new Pyudf_il_i());
    opTab.addOperator(new Pyudf_ii_l());
    opTab.addOperator(new Pyudf_i_f());
    opTab.addOperator(new Pyudf_f_i());
    opTab.addOperator(new Pyudf_ii_f());
    opTab.addOperator(new Pyudf_if_i());
    opTab.addOperator(new Pyudf_if_f());
    opTab.addOperator(new Pyudf_fi_i());
    opTab.addOperator(new Pyudf_fi_f());
    opTab.addOperator(new Pyudf_ff_i());
    ////MAPD_UDF1END-DONOTCHANGETHISCOMMENT
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

    ////MAPD_UDF2START-DONOTCHANGETHISCOMMENT
    /* Code within this MAPD_UDF2 block is generated by mapd_udf.py script */

public static class Pyudf_d_d extends SqlFunction {
  Pyudf_d_d() {
    super("PYUDF_D_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dd_d extends SqlFunction {
  Pyudf_dd_d() {
    super("PYUDF_DD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_ddd_d extends SqlFunction {
  Pyudf_ddd_d() {
    super("PYUDF_DDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (3+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dddd_d extends SqlFunction {
  Pyudf_dddd_d() {
    super("PYUDF_DDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (4+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_ddddd_d extends SqlFunction {
  Pyudf_ddddd_d() {
    super("PYUDF_DDDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (5+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dddddd_d extends SqlFunction {
  Pyudf_dddddd_d() {
    super("PYUDF_DDDDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (6+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_ddddddd_d extends SqlFunction {
  Pyudf_ddddddd_d() {
    super("PYUDF_DDDDDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (7+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dddddddd_d extends SqlFunction {
  Pyudf_dddddddd_d() {
    super("PYUDF_DDDDDDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (8+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_ddddddddd_d extends SqlFunction {
  Pyudf_ddddddddd_d() {
    super("PYUDF_DDDDDDDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (9+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dddddddddd_d extends SqlFunction {
  Pyudf_dddddddddd_d() {
    super("PYUDF_DDDDDDDDDD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (10+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_f_f extends SqlFunction {
  Pyudf_f_f() {
    super("PYUDF_F_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ff_f extends SqlFunction {
  Pyudf_ff_f() {
    super("PYUDF_FF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fff_f extends SqlFunction {
  Pyudf_fff_f() {
    super("PYUDF_FFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (3+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ffff_f extends SqlFunction {
  Pyudf_ffff_f() {
    super("PYUDF_FFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (4+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fffff_f extends SqlFunction {
  Pyudf_fffff_f() {
    super("PYUDF_FFFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (5+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ffffff_f extends SqlFunction {
  Pyudf_ffffff_f() {
    super("PYUDF_FFFFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (6+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fffffff_f extends SqlFunction {
  Pyudf_fffffff_f() {
    super("PYUDF_FFFFFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (7+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ffffffff_f extends SqlFunction {
  Pyudf_ffffffff_f() {
    super("PYUDF_FFFFFFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (8+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fffffffff_f extends SqlFunction {
  Pyudf_fffffffff_f() {
    super("PYUDF_FFFFFFFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (9+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ffffffffff_f extends SqlFunction {
  Pyudf_ffffffffff_f() {
    super("PYUDF_FFFFFFFFFF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (10+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_l_l extends SqlFunction {
  Pyudf_l_l() {
    super("PYUDF_L_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_ll_l extends SqlFunction {
  Pyudf_ll_l() {
    super("PYUDF_LL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_lll_l extends SqlFunction {
  Pyudf_lll_l() {
    super("PYUDF_LLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (3+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_llll_l extends SqlFunction {
  Pyudf_llll_l() {
    super("PYUDF_LLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (4+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_lllll_l extends SqlFunction {
  Pyudf_lllll_l() {
    super("PYUDF_LLLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (5+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_llllll_l extends SqlFunction {
  Pyudf_llllll_l() {
    super("PYUDF_LLLLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (6+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_lllllll_l extends SqlFunction {
  Pyudf_lllllll_l() {
    super("PYUDF_LLLLLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (7+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_llllllll_l extends SqlFunction {
  Pyudf_llllllll_l() {
    super("PYUDF_LLLLLLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (8+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_lllllllll_l extends SqlFunction {
  Pyudf_lllllllll_l() {
    super("PYUDF_LLLLLLLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (9+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_llllllllll_l extends SqlFunction {
  Pyudf_llllllllll_l() {
    super("PYUDF_LLLLLLLLLL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (10+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_i_i extends SqlFunction {
  Pyudf_i_i() {
    super("PYUDF_I_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_ii_i extends SqlFunction {
  Pyudf_ii_i() {
    super("PYUDF_II_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iii_i extends SqlFunction {
  Pyudf_iii_i() {
    super("PYUDF_III_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (3+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiii_i extends SqlFunction {
  Pyudf_iiii_i() {
    super("PYUDF_IIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (4+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiiii_i extends SqlFunction {
  Pyudf_iiiii_i() {
    super("PYUDF_IIIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (5+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiiiii_i extends SqlFunction {
  Pyudf_iiiiii_i() {
    super("PYUDF_IIIIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (6+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiiiiii_i extends SqlFunction {
  Pyudf_iiiiiii_i() {
    super("PYUDF_IIIIIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (7+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiiiiiii_i extends SqlFunction {
  Pyudf_iiiiiiii_i() {
    super("PYUDF_IIIIIIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (8+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiiiiiiii_i extends SqlFunction {
  Pyudf_iiiiiiiii_i() {
    super("PYUDF_IIIIIIIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (9+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_iiiiiiiiii_i extends SqlFunction {
  Pyudf_iiiiiiiiii_i() {
    super("PYUDF_IIIIIIIIII_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (10+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_d_f extends SqlFunction {
  Pyudf_d_f() {
    super("PYUDF_D_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_f_d extends SqlFunction {
  Pyudf_f_d() {
    super("PYUDF_F_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dd_f extends SqlFunction {
  Pyudf_dd_f() {
    super("PYUDF_DD_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_df_d extends SqlFunction {
  Pyudf_df_d() {
    super("PYUDF_DF_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_df_f extends SqlFunction {
  Pyudf_df_f() {
    super("PYUDF_DF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fd_d extends SqlFunction {
  Pyudf_fd_d() {
    super("PYUDF_FD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_fd_f extends SqlFunction {
  Pyudf_fd_f() {
    super("PYUDF_FD_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ff_d extends SqlFunction {
  Pyudf_ff_d() {
    super("PYUDF_FF_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_d_l extends SqlFunction {
  Pyudf_d_l() {
    super("PYUDF_D_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_l_d extends SqlFunction {
  Pyudf_l_d() {
    super("PYUDF_L_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dd_l extends SqlFunction {
  Pyudf_dd_l() {
    super("PYUDF_DD_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_dl_d extends SqlFunction {
  Pyudf_dl_d() {
    super("PYUDF_DL_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dl_l extends SqlFunction {
  Pyudf_dl_l() {
    super("PYUDF_DL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_ld_d extends SqlFunction {
  Pyudf_ld_d() {
    super("PYUDF_LD_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_ld_l extends SqlFunction {
  Pyudf_ld_l() {
    super("PYUDF_LD_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_ll_d extends SqlFunction {
  Pyudf_ll_d() {
    super("PYUDF_LL_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_d_i extends SqlFunction {
  Pyudf_d_i() {
    super("PYUDF_D_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_i_d extends SqlFunction {
  Pyudf_i_d() {
    super("PYUDF_I_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_dd_i extends SqlFunction {
  Pyudf_dd_i() {
    super("PYUDF_DD_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_di_d extends SqlFunction {
  Pyudf_di_d() {
    super("PYUDF_DI_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_di_i extends SqlFunction {
  Pyudf_di_i() {
    super("PYUDF_DI_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_id_d extends SqlFunction {
  Pyudf_id_d() {
    super("PYUDF_ID_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_id_i extends SqlFunction {
  Pyudf_id_i() {
    super("PYUDF_ID_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_ii_d extends SqlFunction {
  Pyudf_ii_d() {
    super("PYUDF_II_D",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }
}

public static class Pyudf_l_f extends SqlFunction {
  Pyudf_l_f() {
    super("PYUDF_L_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_f_l extends SqlFunction {
  Pyudf_f_l() {
    super("PYUDF_F_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_ll_f extends SqlFunction {
  Pyudf_ll_f() {
    super("PYUDF_LL_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_lf_l extends SqlFunction {
  Pyudf_lf_l() {
    super("PYUDF_LF_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_lf_f extends SqlFunction {
  Pyudf_lf_f() {
    super("PYUDF_LF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fl_l extends SqlFunction {
  Pyudf_fl_l() {
    super("PYUDF_FL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_fl_f extends SqlFunction {
  Pyudf_fl_f() {
    super("PYUDF_FL_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ff_l extends SqlFunction {
  Pyudf_ff_l() {
    super("PYUDF_FF_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_l_i extends SqlFunction {
  Pyudf_l_i() {
    super("PYUDF_L_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_i_l extends SqlFunction {
  Pyudf_i_l() {
    super("PYUDF_I_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_ll_i extends SqlFunction {
  Pyudf_ll_i() {
    super("PYUDF_LL_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_li_l extends SqlFunction {
  Pyudf_li_l() {
    super("PYUDF_LI_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_li_i extends SqlFunction {
  Pyudf_li_i() {
    super("PYUDF_LI_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_il_l extends SqlFunction {
  Pyudf_il_l() {
    super("PYUDF_IL_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_il_i extends SqlFunction {
  Pyudf_il_i() {
    super("PYUDF_IL_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_ii_l extends SqlFunction {
  Pyudf_ii_l() {
    super("PYUDF_II_L",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.BIGINT);
  }
}

public static class Pyudf_i_f extends SqlFunction {
  Pyudf_i_f() {
    super("PYUDF_I_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_f_i extends SqlFunction {
  Pyudf_f_i() {
    super("PYUDF_F_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (1+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_ii_f extends SqlFunction {
  Pyudf_ii_f() {
    super("PYUDF_II_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_if_i extends SqlFunction {
  Pyudf_if_i() {
    super("PYUDF_IF_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_if_f extends SqlFunction {
  Pyudf_if_f() {
    super("PYUDF_IF_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_fi_i extends SqlFunction {
  Pyudf_fi_i() {
    super("PYUDF_FI_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}

public static class Pyudf_fi_f extends SqlFunction {
  Pyudf_fi_f() {
    super("PYUDF_FI_F",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.FLOAT);
  }
}

public static class Pyudf_ff_i extends SqlFunction {
  Pyudf_ff_i() {
    super("PYUDF_FF_I",
      SqlKind.OTHER_FUNCTION,
      null,
      null,
      OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
      SqlFunctionCategory.SYSTEM);
  }
  @Override
  public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
    assert opBinding.getOperandCount() == (2+1);
    final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
    return typeFactory.createSqlType(SqlTypeName.INTEGER);
  }
}
    ////MAPD_UDF2END-DONOTCHANGETHISCOMMENT

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
      return typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
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
              typeFactory.createSqlType(SqlTypeName.TIMESTAMP),
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
              typeFactory.createSqlType(SqlTypeName.TIMESTAMP),
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

    private static class EscapeOptional implements Predicate<Integer> {
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

    private static class EscapeOptional implements Predicate<Integer> {
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
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
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
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_intersects_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_intersects_sig.add(SqlTypeFamily.ANY);
      st_intersects_sig.add(SqlTypeFamily.ANY);
      return st_intersects_sig;
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
      return typeFactory.createSqlType(SqlTypeName.BOOLEAN);
    }

    private static java.util.List<SqlTypeFamily> signature() {
      java.util.List<SqlTypeFamily> st_within_sig =
              new java.util.ArrayList<SqlTypeFamily>();
      st_within_sig.add(SqlTypeFamily.ANY);
      st_within_sig.add(SqlTypeFamily.ANY);
      return st_within_sig;
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.DOUBLE);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
      return typeFactory.createSqlType(SqlTypeName.INTEGER);
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
              SqlFunctionCategory.SYSTEM);
      ret = toSqlTypeName(sig.getRet());
    }

    private static java.util.List<SqlTypeFamily> toSqlSignature(
            final ExtensionFunction sig) {
      java.util.List<SqlTypeFamily> sql_sig = new java.util.ArrayList<SqlTypeFamily>();
      for (int arg_idx = 0; arg_idx < sig.getArgs().size(); ++arg_idx) {
        final ExtensionFunction.ExtArgumentType arg_type = sig.getArgs().get(arg_idx);
        sql_sig.add(toSqlTypeName(arg_type).getFamily());
        if (isPointerType(arg_type)) {
          ++arg_idx;
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
      final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
      return typeFactory.createTypeWithNullability(typeFactory.createSqlType(ret), true);
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
          return SqlTypeName.ARRAY;
      }
      assert false;
      return null;
    }

    private final SqlTypeName ret;
  }

  //
  // Internal accessors for in-situ poly render queries
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
}

// End MapDSqlOperatorTable.java

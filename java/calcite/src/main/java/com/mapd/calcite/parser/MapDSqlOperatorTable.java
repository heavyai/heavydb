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

import java.util.HashSet;
import java.util.Map;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
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
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.util.ListSqlOperatorTable;

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

    /**
     * Mock operator table for testing purposes. Contains the standard SQL
     * operator table, plus a list of operators.
     */
    //~ Instance fields --------------------------------------------------------
    private final ListSqlOperatorTable listOpTab;

    //~ Constructors -----------------------------------------------------------
    public MapDSqlOperatorTable(SqlOperatorTable parentTable) {
        super(ImmutableList.of(parentTable, new CaseInsensitiveListSqlOperatorTable()));
        listOpTab = (ListSqlOperatorTable) tableList.get(1);
    }

    //~ Methods ----------------------------------------------------------------
    /**
     * Adds an operator to this table.
     *
     * @param op
     */
    public void addOperator(SqlOperator op) {
        listOpTab.add(op);
    }

    
    public static void addUDF(MapDSqlOperatorTable opTab, final Map<String, ExtensionFunction> extSigs) {
        // Don't use anonymous inner classes. They can't be instantiated
        // using reflection when we are deserializing from JSON.
        //opTab.addOperator(new RampFunction());
        //opTab.addOperator(new DedupFunction());
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
        opTab.addOperator(new ApproxCountDistinct());
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

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.builder()
                    .add("NAME", SqlTypeName.VARCHAR, 1024)
                    .build();
        }
    }

    /**
     * "MyUDFFunction" user-defined function test. our udf's will look like
     * system functions to calcite as it has no access to the code
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.BIGINT);
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createTypeWithNullability(typeFactory.createSqlType(SqlTypeName.BIGINT),
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createTypeWithNullability(typeFactory.createSqlType(SqlTypeName.BIGINT),
                    opBinding.getOperandType(1).isNullable());
        }
    }

    public static class Dateadd extends SqlFunction {

        public Dateadd() {
            super("DATEADD",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.DATETIME),
                    SqlFunctionCategory.TIMEDATE);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createTypeWithNullability(typeFactory.createSqlType(SqlTypeName.TIMESTAMP),
                    opBinding.getOperandType(2).isNullable());
        }
    }

    public static class Datediff extends SqlFunction {

        public Datediff() {
            super("DATEDIFF",
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.DATETIME, SqlTypeFamily.DATETIME),
                    SqlFunctionCategory.TIMEDATE);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createTypeWithNullability(typeFactory.createSqlType(SqlTypeName.BIGINT),
                    opBinding.getOperandType(1).isNullable() || opBinding.getOperandType(2).isNullable());
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createTypeWithNullability(typeFactory.createSqlType(SqlTypeName.TIMESTAMP),
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
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
            java.util.ArrayList<SqlTypeFamily> families = new java.util.ArrayList<SqlTypeFamily>();
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
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
            java.util.ArrayList<SqlTypeFamily> families = new java.util.ArrayList<SqlTypeFamily>();
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
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
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
            super("SIGN", SqlKind.OTHER_FUNCTION, null, null, OperandTypes.NUMERIC, SqlFunctionCategory.NUMERIC);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            return opBinding.getOperandType(0);
        }
    }

    static class Truncate extends SqlFunction {

        Truncate() {
            super("TRUNCATE", SqlKind.OTHER_FUNCTION, null, null, OperandTypes.family(signature()), SqlFunctionCategory.NUMERIC);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            assert opBinding.getOperandCount() == 2;
            return opBinding.getOperandType(0);
        }

        private static java.util.List<SqlTypeFamily> signature() {
            java.util.List<SqlTypeFamily> truncate_sig = new java.util.ArrayList<SqlTypeFamily>();
            truncate_sig.add(SqlTypeFamily.NUMERIC);
            truncate_sig.add(SqlTypeFamily.INTEGER);
            return truncate_sig;
        }
    }

    static class ApproxCountDistinct extends SqlAggFunction {

        ApproxCountDistinct() {
            super("APPROX_COUNT_DISTINCT",
                    null,
                    SqlKind.OTHER_FUNCTION,
                    null,
                    null,
                    OperandTypes.or(
                            OperandTypes.family(SqlTypeFamily.ANY),
                            OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.INTEGER)),
                    SqlFunctionCategory.SYSTEM);
        }

        @Override
        public RelDataType inferReturnType(SqlOperatorBinding opBinding) {
            final RelDataTypeFactory typeFactory
                    = opBinding.getTypeFactory();
            return typeFactory.createSqlType(SqlTypeName.BIGINT);
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

        private static java.util.List<SqlTypeFamily> toSqlSignature(final ExtensionFunction sig) {
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
        return type == ExtensionFunction.ExtArgumentType.PInt16
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

        private static SqlTypeName toSqlTypeName(final ExtensionFunction.ExtArgumentType type) {
            switch (type) {
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
}

// End MapDSqlOperatorTable.java

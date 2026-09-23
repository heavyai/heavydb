/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// HeavyDBSqlToRelConverter was intended to factor out all HeavyDB-specific logic from the
// vendored Calcite SqlToRelConverter into this subclass, but that refactor was never completed.
// HeavyDB-specific modifications still exist directly in the vendored SqlToRelConverter source.
//
// What has been factored out here:
//   - TRY_CAST conversion: registers a custom SqlRexConvertlet for the HeavyDB TRY_CAST operator
//     that mirrors the standard CAST conversion but preserves the TRY_CAST call in the Rex tree.
//   - getColumnMappings() override: handles ExtTableFunction (HeavyDB C++ UDTFs), which use
//     ReturnTypes.CURSOR rather than TableFunctionReturnTypeInference. The override delegates to
//     ExtTableFunction.getColumnMappings() directly so that filter pushdown into CURSOR arguments
//     still works for these operators.
//   - CONFIG / Config interface: a HeavyDB-specific default SqlToRelConverter.Config that enables
//     JSON type operators, sets the logical RelBuilder, and exposes an expandPredicate hook. The
//     inner Config interface re-declares all "with*" builder methods so that the Immutables-generated
//     builder returns HeavyDBSqlToRelConverter.Config rather than the base SqlToRelConverter.Config.
package org.apache.calcite.sql2rel;

import com.google.common.collect.ImmutableList;
import com.mapd.calcite.parser.HeavyDBSqlOperatorTable;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.metadata.RelColumnMapping;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.immutables.value.Value;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.hint.HintStrategyTable;

import java.util.Set;
import java.util.function.BiPredicate;
@Value.Enclosing
public class HeavyDBSqlToRelConverter extends SqlToRelConverter {

    private final static BiPredicate<SqlNode, SqlNode> defaultPredicate = new BiPredicate<SqlNode, SqlNode>() {
        @Override
        public boolean test(SqlNode root, SqlNode expression) {
            throw new RuntimeException("Test for default BiPredicate not implemented");
        }
    };
    public static final Config CONFIG = ImmutableHeavyDBSqlToRelConverter.Config.builder()
          .withHintStrategyTable(HintStrategyTable.EMPTY)
          .withRelBuilderFactory(RelFactories.LOGICAL_BUILDER)
          .withRelBuilderConfigTransform(c -> c.withPushJoinCondition(true))
          .withHintStrategyTable(HintStrategyTable.EMPTY)
          .withAddJsonTypeOperatorEnabled(true)
          .withExpandPredicate(defaultPredicate).build();

    public HeavyDBSqlToRelConverter(
            RelOptTable.ViewExpander viewExpander,
            @Nullable SqlValidator validator,
            Prepare.CatalogReader catalogReader,
            RelOptCluster cluster,
            SqlRexConvertletTable convertletTable,
            SqlToRelConverter.Config config) {
        super(viewExpander, validator, catalogReader, cluster, convertletTable, config);
        StandardConvertletTable.INSTANCE.registerOp(HeavyDBSqlOperatorTable.TRY_CAST, this::convertTryCast);
    }

    protected RexNode convertTryCast(SqlRexContext cx, final SqlCall call) {
        RelDataTypeFactory typeFactory = cx.getTypeFactory();
        // assert call.getKind() == SqlKind.CAST;
        final SqlNode left = call.operand(0);
        final SqlNode right = call.operand(1);

        SqlDataTypeSpec dataType = (SqlDataTypeSpec) right;
        RelDataType type = dataType.deriveType(cx.getValidator());
        if (type == null) {
            type = cx.getValidator().getValidatedNodeType(dataType.getTypeName());
        }
        RexNode arg = cx.convertExpression(left);
        if (arg.getType().isNullable()) {
            type = typeFactory.createTypeWithNullability(type, true);
        }
        if (SqlUtil.isNullLiteral(left, false)) {
            final SqlValidatorImpl validator = (SqlValidatorImpl) cx.getValidator();
            validator.setValidatedNodeType(left, type);
            return cx.convertExpression(left);
        }
        return cx.getRexBuilder().makeCall(
                type, HeavyDBSqlOperatorTable.TRY_CAST, ImmutableList.of(arg));
    }

    @Override
    protected @Nullable Set<RelColumnMapping> getColumnMappings(SqlOperator op) {
        // ExtTableFunction represents HeavyDB user-defined table functions (UDTFs),
        // registered from C++ headers. These use ReturnTypes.CURSOR rather than
        // TableFunctionReturnTypeInference, so the base class check for column
        // mappings fails. We call getColumnMappings() directly on the ExtTableFunction
        // to enable filter pushdown into CURSOR arguments.
        if (op instanceof HeavyDBSqlOperatorTable.ExtTableFunction) {
            return ((HeavyDBSqlOperatorTable.ExtTableFunction) op).getColumnMappings();
        }
        return super.getColumnMappings(op);
    }

    public static Config config() {return CONFIG;}
   // The combination of @Value.Enclosing and @Value.Immutable generates ImmutableHeavyDBSqlToRelConverter.Config
    @Value.Immutable(singleton = false)
    public interface Config extends org.apache.calcite.sql2rel.SqlToRelConverter.Config {
        // These overridden methods, copied from the calcite SqlToRelConverter class are required
        // so that calls to the "with*" methods call those from the class
        // automatically generated from this code via the @Value.Immutable annotation,
        // rather than its superclass.
        @Override Config withDecorrelationEnabled(boolean decorrelationEnabled);
        @Override Config withTrimUnusedFields(boolean trimUnusedFields);
        @Override Config withCreateValuesRel(boolean createValuesRel);
        @Override Config withExplain(boolean explain);
        @Override Config withExpand(boolean expand);
        @Override Config withInSubQueryThreshold(int threshold);
        @Override Config withRemoveSortInSubQuery(boolean removeSortInSubQuery);
        @Override Config withRelBuilderFactory(RelBuilderFactory factory);
        @Override Config withRelBuilderConfigTransform(java.util.function.UnaryOperator<RelBuilder.Config> transform);
        @Override Config withHintStrategyTable(HintStrategyTable hintStrategyTable);
        @Override Config withAddJsonTypeOperatorEnabled(boolean addJsonTypeOperatorEnabled);
        Config withExpandPredicate(BiPredicate<SqlNode, SqlNode> bipredicate);
        BiPredicate<SqlNode, SqlNode> getExpandPredicate();
    }
}

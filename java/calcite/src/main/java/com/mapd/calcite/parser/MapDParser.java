/*
 * Copyright 2019 OmniSci, Inc.
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

import static org.apache.calcite.sql.parser.SqlParserPos.ZERO;

import com.google.common.collect.ImmutableList;
import com.mapd.common.SockTransportProperties;
import com.mapd.parser.server.ExtensionFunction;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.config.CalciteConnectionConfigImpl;
import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptLattice;
import org.apache.calcite.plan.RelOptMaterialization;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.MapDPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.RelShuttleImpl;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.TableModify.Operation;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalTableModify;
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider;
import org.apache.calcite.rel.rules.FilterMergeRule;
import org.apache.calcite.rel.rules.FilterProjectTransposeRule;
import org.apache.calcite.rel.rules.JoinProjectTransposeRule;
import org.apache.calcite.rel.rules.ProjectMergeRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.JoinConditionType;
import org.apache.calcite.sql.JoinType;
import org.apache.calcite.sql.SqlAsOperator;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCollectionTypeNameSpec;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlJoin;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.SqlUnresolvedFunction;
import org.apache.calcite.sql.SqlUpdate;
import org.apache.calcite.sql.fun.SqlCastFunction;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.util.SqlBasicVisitor;
import org.apache.calcite.sql.util.SqlShuttle;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.apache.calcite.util.ConversionUtil;
import org.apache.calcite.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiPredicate;

/**
 *
 * @author michael
 */
public final class MapDParser {
  public static final ThreadLocal<MapDParser> CURRENT_PARSER = new ThreadLocal<>();
  private static final EnumSet<SqlKind> SCALAR =
          EnumSet.of(SqlKind.SCALAR_QUERY, SqlKind.SELECT);
  private static final EnumSet<SqlKind> EXISTS = EnumSet.of(SqlKind.EXISTS);
  private static final EnumSet<SqlKind> DELETE = EnumSet.of(SqlKind.DELETE);
  private static final EnumSet<SqlKind> UPDATE = EnumSet.of(SqlKind.UPDATE);
  private static final EnumSet<SqlKind> IN = EnumSet.of(SqlKind.IN);

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDParser.class);

  // private SqlTypeFactoryImpl typeFactory;
  // private MapDCatalogReader catalogReader;
  // private SqlValidatorImpl validator;
  // private SqlToRelConverter converter;
  private final Map<String, ExtensionFunction> extSigs;
  private final String dataDir;

  private int callCount = 0;
  private final int mapdPort;
  private MapDUser mapdUser;
  SqlNode sqlNode_;
  private SockTransportProperties sock_transport_properties = null;

  private static Map<String, Boolean> SubqueryCorrMemo = new ConcurrentHashMap<>();

  public MapDParser(String dataDir,
          final Map<String, ExtensionFunction> extSigs,
          int mapdPort,
          SockTransportProperties skT) {
    System.setProperty(
            "saffron.default.charset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty(
            "saffron.default.nationalcharset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty("saffron.default.collation.name",
            ConversionUtil.NATIVE_UTF16_CHARSET_NAME + "$en_US");
    this.dataDir = dataDir;
    this.extSigs = extSigs;
    this.mapdPort = mapdPort;
    this.sock_transport_properties = skT;
  }

  public void clearMemo() {
    SubqueryCorrMemo.clear();
  }

  private static final Context MAPD_CONNECTION_CONTEXT = new Context() {
    MapDRelDataTypeSystemImpl myTypeSystem = new MapDRelDataTypeSystemImpl();
    CalciteConnectionConfig config = new CalciteConnectionConfigImpl(new Properties()) {
      @SuppressWarnings("unchecked")
      public <T extends Object> T typeSystem(
              java.lang.Class<T> typeSystemClass, T defaultTypeSystem) {
        return (T) myTypeSystem;
      };

      public boolean caseSensitive() {
        return false;
      };

      public org.apache.calcite.sql.validate.SqlConformance conformance() {
        return SqlConformanceEnum.LENIENT;
      };
    };

    @Override
    public <C> C unwrap(Class<C> aClass) {
      if (aClass.isInstance(config)) {
        return aClass.cast(config);
      }
      return null;
    }
  };

  private MapDPlanner getPlanner() {
    return getPlanner(true, true);
  }

  private boolean isCorrelated(SqlNode expression) {
    String queryString = expression.toSqlString(SqlDialect.CALCITE).getSql();
    Boolean isCorrelatedSubquery = SubqueryCorrMemo.get(queryString);
    if (null != isCorrelatedSubquery) {
      return isCorrelatedSubquery;
    }

    try {
      MapDParser parser =
              new MapDParser(dataDir, extSigs, mapdPort, sock_transport_properties);
      MapDParserOptions options = new MapDParserOptions();
      parser.setUser(mapdUser);
      parser.getRelAlgebra(expression.toSqlString(SqlDialect.CALCITE).getSql(), options);
    } catch (Exception e) {
      // if we are not able to parse, then assume correlated
      SubqueryCorrMemo.put(queryString, true);
      return true;
    }
    SubqueryCorrMemo.put(queryString, false);
    return false;
  }

  private MapDPlanner getPlanner(final boolean allowSubQueryExpansion,
          final boolean allowPushdownJoinCondition) {
    final MapDSchema mapd =
            new MapDSchema(dataDir, this, mapdPort, mapdUser, sock_transport_properties);
    final SchemaPlus rootSchema = Frameworks.createRootSchema(true);

    BiPredicate<SqlNode, SqlNode> expandPredicate = new BiPredicate<SqlNode, SqlNode>() {
      @Override
      public boolean test(SqlNode root, SqlNode expression) {
        if (!allowSubQueryExpansion) {
          return false;
        }

        // special handling of sub-queries
        if (expression.isA(SCALAR) || expression.isA(EXISTS) || expression.isA(IN)) {
          // only expand if it is correlated.

          if (expression.isA(EXISTS)) {
            // always expand subquery by EXISTS clause
            return true;
          }

          if (expression.isA(IN)) {
            // expand subquery by IN clause
            // but correlated subquery by NOT_IN clause is not available
            // currently due to a lack of supporting in Calcite
            boolean found_expression = false;
            if (expression instanceof SqlCall) {
              SqlCall call = (SqlCall) expression;
              if (call.getOperandList().size() == 2) {
                // if IN clause is correlated, its second operand of corresponding
                // expression is SELECT clause which indicates a correlated subquery.
                // Here, an expression "f.val IN (SELECT ...)" has two operands.
                // Since we have interest in its subquery, so try to check whether
                // the second operand, i.e., call.getOperandList().get(1)
                // is a type of SqlSelect and also is correlated.
                // Note that the second operand of non-correlated IN clause
                // does not have SqlSelect as its second operand
                if (call.getOperandList().get(1) instanceof SqlSelect) {
                  expression = call.getOperandList().get(1);
                  SqlSelect select_call = (SqlSelect) expression;
                  if (select_call.hasWhere()) {
                    found_expression = true;
                  }
                }
              }
            }
            if (!found_expression) {
              return false;
            }
          }

          if (isCorrelated(expression)) {
            SqlSelect select = null;
            if (expression instanceof SqlCall) {
              SqlCall call = (SqlCall) expression;
              if (call.getOperator().equals(SqlStdOperatorTable.SCALAR_QUERY)) {
                expression = call.getOperandList().get(0);
              }
            }

            if (expression instanceof SqlSelect) {
              select = (SqlSelect) expression;
            }

            if (null != select) {
              if (null != select.getFetch() || null != select.getOffset()
                      || (null != select.getOrderList()
                              && select.getOrderList().size() != 0)) {
                throw new CalciteException(
                        "Correlated sub-queries with ordering not supported.", null);
              }
            }
            return true;
          }
        }

        // per default we do not want to expand
        return false;
      }
    };

    BiPredicate<SqlNode, Join> pushdownJoinPredicate = new BiPredicate<SqlNode, Join>() {
      @Override
      public boolean test(SqlNode t, Join u) {
        if (!allowPushdownJoinCondition) {
          return false;
        }

        return !hasGeoColumns(u.getRowType());
      }

      private boolean hasGeoColumns(RelDataType type) {
        for (RelDataTypeField f : type.getFieldList()) {
          if ("any".equalsIgnoreCase(f.getType().getFamily().toString())) {
            // any indicates geo types at the moment
            return true;
          }
        }

        return false;
      }
    };

    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(rootSchema.add(mapdUser.getDB(), mapd))
                    .operatorTable(createOperatorTable(extSigs))
                    .parserConfig(SqlParser.configBuilder()
                                          .setConformance(SqlConformanceEnum.LENIENT)
                                          .setUnquotedCasing(Casing.UNCHANGED)
                                          .setCaseSensitive(false)
                                          .build())
                    .sqlToRelConverterConfig(
                            SqlToRelConverter
                                    .configBuilder()
                                    // enable sub-query expansion (de-correlation)
                                    .withExpandPredicate(expandPredicate)
                                    // allow as many as possible IN operator values
                                    .withInSubQueryThreshold(Integer.MAX_VALUE)
                                    .withPushdownJoinCondition(pushdownJoinPredicate)
                                    .build())
                    .typeSystem(createTypeSystem())
                    .context(MAPD_CONNECTION_CONTEXT)
                    .build();
    return new MapDPlanner(config);
  }

  public void setUser(MapDUser mapdUser) {
    this.mapdUser = mapdUser;
  }

  public String getRelAlgebra(String sql, final MapDParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    callCount++;
    final RelRoot sqlRel = queryToSqlNode(sql, parserOptions);

    RelNode project = sqlRel.project();

    if (parserOptions.isExplain()) {
      return RelOptUtil.toString(sqlRel.project());
    }

    String res = MapDSerializer.toString(project);

    return res;
  }

  public MapDPlanner.CompletionResult getCompletionHints(
          String sql, int cursor, List<String> visible_tables) {
    return getPlanner().getCompletionHints(sql, cursor, visible_tables);
  }

  public Set<String> resolveSelectIdentifiers(SqlIdentifierCapturer capturer) {
    MapDSchema schema =
            new MapDSchema(dataDir, this, mapdPort, mapdUser, sock_transport_properties);
    HashSet<String> resolved = new HashSet<>();

    for (String name : capturer.selects) {
      MapDTable table = (MapDTable) schema.getTable(name);
      if (null == table) {
        throw new RuntimeException("table/view not found: " + name);
      }

      if (table instanceof MapDView) {
        MapDView view = (MapDView) table;
        resolved.addAll(resolveSelectIdentifiers(view.getAccessedObjects()));
      } else {
        resolved.add(name);
      }
    }

    return resolved;
  }

  private String getTableName(SqlNode node) {
    if (node.isA(EnumSet.of(SqlKind.AS))) {
      node = ((SqlCall) node).getOperandList().get(1);
    }
    if (node instanceof SqlIdentifier) {
      SqlIdentifier id = (SqlIdentifier) node;
      return id.names.get(id.names.size() - 1);
    }
    return null;
  }

  private SqlSelect rewriteSimpleUpdateAsSelect(final SqlUpdate update) {
    SqlNode where = update.getCondition();

    if (update.getSourceExpressionList().size() != 1) {
      return null;
    }

    if (!(update.getSourceExpressionList().get(0) instanceof SqlSelect)) {
      return null;
    }

    final SqlSelect inner = (SqlSelect) update.getSourceExpressionList().get(0);

    if (null != inner.getGroup() || null != inner.getFetch() || null != inner.getOffset()
            || (null != inner.getOrderList() && inner.getOrderList().size() != 0)
            || (null != inner.getGroup() && inner.getGroup().size() != 0)
            || null == getTableName(inner.getFrom())) {
      return null;
    }

    if (!isCorrelated(inner)) {
      return null;
    }

    final String updateTableName = getTableName(update.getTargetTable());

    if (null != where) {
      where = where.accept(new SqlShuttle() {
        @Override
        public SqlNode visit(SqlIdentifier id) {
          if (id.isSimple()) {
            id = new SqlIdentifier(Arrays.asList(updateTableName, id.getSimple()),
                    id.getParserPosition());
          }

          return id;
        }
      });
    }

    SqlJoin join = new SqlJoin(ZERO,
            update.getTargetTable(),
            SqlLiteral.createBoolean(false, ZERO),
            SqlLiteral.createSymbol(JoinType.LEFT, ZERO),
            inner.getFrom(),
            SqlLiteral.createSymbol(JoinConditionType.ON, ZERO),
            inner.getWhere());

    SqlNode select0 = inner.getSelectList().get(0);

    boolean wrapInSingleValue = true;
    if (select0 instanceof SqlCall) {
      SqlCall selectExprCall = (SqlCall) select0;
      if (Util.isSingleValue(selectExprCall)) {
        wrapInSingleValue = false;
      }
    }

    if (wrapInSingleValue) {
      select0 = new SqlBasicCall(
              SqlStdOperatorTable.SINGLE_VALUE, new SqlNode[] {select0}, ZERO);
    }

    SqlNodeList selectList = new SqlNodeList(ZERO);
    selectList.add(select0);
    selectList.add(new SqlBasicCall(SqlStdOperatorTable.AS,
            new SqlNode[] {new SqlBasicCall(
                                   new SqlUnresolvedFunction(
                                           new SqlIdentifier("OFFSET_IN_FRAGMENT", ZERO),
                                           null,
                                           null,
                                           null,
                                           null,
                                           SqlFunctionCategory.USER_DEFINED_FUNCTION),
                                   new SqlNode[0],
                                   SqlParserPos.ZERO),
                    new SqlIdentifier("EXPR$DELETE_OFFSET_IN_FRAGMENT", ZERO)},
            ZERO));

    SqlNodeList groupBy = new SqlNodeList(ZERO);
    groupBy.add(new SqlIdentifier("EXPR$DELETE_OFFSET_IN_FRAGMENT", ZERO));

    SqlSelect select = new SqlSelect(
            ZERO, null, selectList, join, where, groupBy, null, null, null, null, null);
    return select;
  }

  private LogicalTableModify getDummyUpdate(SqlUpdate update)
          throws SqlParseException, ValidationException, RelConversionException {
    SqlIdentifier targetTable = (SqlIdentifier) update.getTargetTable();
    String targetTableName = targetTable.names.get(targetTable.names.size() - 1);
    MapDPlanner planner = getPlanner();
    String dummySql = "UPDATE " + targetTableName + " SET "
            + update.getTargetColumnList()
                      .get(0)
                      .toSqlString(SqlDialect.CALCITE)
                      .toString()
            + " = NULL";
    SqlNode dummyNode = planner.parse(dummySql);
    dummyNode = planner.validate(dummyNode);
    RelRoot dummyRoot = planner.rel(dummyNode);
    LogicalTableModify dummyModify = (LogicalTableModify) dummyRoot.rel;
    return dummyModify;
  }

  private RelRoot rewriteUpdateAsSelect(SqlUpdate update, MapDParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    int correlatedQueriesCount[] = new int[1];
    SqlBasicVisitor<Void> correlatedQueriesCounter = new SqlBasicVisitor<Void>() {
      @Override
      public Void visit(SqlCall call) {
        if (call.isA(SCALAR)) {
          if (isCorrelated(call)) {
            correlatedQueriesCount[0]++;
          }
        }
        return super.visit(call);
      }
    };

    update.accept(correlatedQueriesCounter);
    if (correlatedQueriesCount[0] > 1) {
      throw new CalciteException(
              "UPDATEs with multiple correlated sub-queries not supported.", null);
    }

    boolean allowPushdownJoinCondition = false;
    SqlNodeList sourceExpression = new SqlNodeList(SqlParserPos.ZERO);
    LogicalTableModify dummyModify = getDummyUpdate(update);
    RelOptTable targetTable = dummyModify.getTable();
    RelDataType targetTableType = targetTable.getRowType();

    SqlSelect select = rewriteSimpleUpdateAsSelect(update);
    boolean applyRexCast = null == select;

    if (null == select) {
      for (int i = 0; i < update.getSourceExpressionList().size(); i++) {
        SqlNode targetColumn = update.getTargetColumnList().get(i);
        SqlNode expression = update.getSourceExpressionList().get(i);

        // special handling of NULL values (ie, make it cast to correct type)
        if (expression instanceof SqlLiteral) {
          SqlLiteral identifierExpression = (SqlLiteral) expression;
          if (null == identifierExpression.getValue()) {
            if (!(targetColumn instanceof SqlIdentifier)) {
              throw new RuntimeException("Unknown identifier type!");
            }

            SqlIdentifier id = (SqlIdentifier) targetColumn;
            RelDataType fieldType =
                    targetTableType
                            .getField(id.names.get(id.names.size() - 1), false, false)
                            .getType();
            if (null != fieldType.getComponentType()) {
              // DO NOT CAST to null array,
              // this is currently not supported in the query engine
              throw new RuntimeException("Updating arrays to NULL not supported!");
              //              expression = new SqlBasicCall(new SqlCastFunction(),
              //                      new SqlNode[] {identifierExpression,
              //                              new SqlDataTypeSpec(
              //                                      new SqlCollectionTypeNameSpec(
              //
              //                                              new SqlBasicTypeNameSpec(
              //                                                      fieldType.getComponentType()
              //                                                              .getSqlTypeName(),
              //                                                      fieldType.getPrecision(),
              //                                                      fieldType.getScale(),
              //                                                      null ==
              //                                                      fieldType.getCharset()
              //                                                              ? null
              //                                                              :
              //                                                              fieldType.getCharset()
              //                                                                        .name(),
              //                                                      SqlParserPos.ZERO),
              //                                              fieldType.getSqlTypeName(),
              //                                              SqlParserPos.ZERO),
              //                                      SqlParserPos.ZERO)},
              //                      SqlParserPos.ZERO);
            } else {
              expression = new SqlBasicCall(new SqlCastFunction(),
                      new SqlNode[] {identifierExpression,
                              new SqlDataTypeSpec(
                                      new SqlBasicTypeNameSpec(fieldType.getSqlTypeName(),
                                              fieldType.getPrecision(),
                                              fieldType.getScale(),
                                              null == fieldType.getCharset()
                                                      ? null
                                                      : fieldType.getCharset().name(),
                                              SqlParserPos.ZERO),
                                      null,
                                      SqlParserPos.ZERO)},
                      SqlParserPos.ZERO);
            }
          }
        }
        sourceExpression.add(expression);
      }

      sourceExpression.add(new SqlBasicCall(SqlStdOperatorTable.AS,
              new SqlNode[] {
                      new SqlBasicCall(new SqlUnresolvedFunction(
                                               new SqlIdentifier("OFFSET_IN_FRAGMENT",
                                                       SqlParserPos.ZERO),
                                               null,
                                               null,
                                               null,
                                               null,
                                               SqlFunctionCategory.USER_DEFINED_FUNCTION),
                              new SqlNode[0],
                              SqlParserPos.ZERO),
                      new SqlIdentifier("EXPR$DELETE_OFFSET_IN_FRAGMENT", ZERO)},
              ZERO));

      select = new SqlSelect(SqlParserPos.ZERO,
              null,
              sourceExpression,
              update.getTargetTable(),
              update.getCondition(),
              null,
              null,
              null,
              null,
              null,
              null);
    }

    MapDPlanner planner = getPlanner(true, allowPushdownJoinCondition);
    SqlNode node = planner.parse(select.toSqlString(SqlDialect.CALCITE).getSql());
    node = planner.validate(node);
    RelRoot root = planner.rel(node);
    LogicalProject project = (LogicalProject) root.project();

    ArrayList<String> fields = new ArrayList<String>();
    ArrayList<RexNode> nodes = new ArrayList<RexNode>();
    final RexBuilder builder = new RexBuilder(planner.getTypeFactory());

    for (SqlNode n : update.getTargetColumnList()) {
      if (n instanceof SqlIdentifier) {
        SqlIdentifier id = (SqlIdentifier) n;
        fields.add(id.names.get(id.names.size() - 1));
      } else {
        throw new RuntimeException("Unknown identifier type!");
      }
    }

    int idx = 0;
    for (RexNode n : project.getChildExps()) {
      if (applyRexCast && idx + 1 < project.getChildExps().size()) {
        RelDataType expectedFieldType =
                targetTableType.getField(fields.get(idx), false, false).getType();
        RexNode exp = project.getChildExps().get(idx);
        if (exp.getType().equals(expectedFieldType)
                || EnumSet.of(SqlKind.ARRAY_VALUE_CONSTRUCTOR).contains(exp.getKind())) {
          nodes.add(project.getChildExps().get(idx));
        } else {
          exp = builder.makeCast(expectedFieldType, exp);
          nodes.add(exp);
        }
      } else {
        nodes.add(project.getChildExps().get(idx));
      }

      idx++;
    }

    ArrayList<RexNode> inputs = new ArrayList<RexNode>();
    int n = 0;
    for (int i = 0; i < fields.size(); i++) {
      inputs.add(
              new RexInputRef(n, project.getRowType().getFieldList().get(n).getType()));
      n++;
    }

    fields.add("EXPR$DELETE_OFFSET_IN_FRAGMENT");
    inputs.add(new RexInputRef(n, project.getRowType().getFieldList().get(n).getType()));

    project = project.copy(
            project.getTraitSet(), project.getInput(), nodes, project.getRowType());

    LogicalTableModify modify = LogicalTableModify.create(targetTable,
            dummyModify.getCatalogReader(),
            project,
            Operation.UPDATE,
            fields,
            inputs,
            true);
    return RelRoot.of(modify, SqlKind.UPDATE);
  }

  RelRoot queryToSqlNode(final String sql, final MapDParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    boolean allowCorrelatedSubQueryExpansion = true;
    boolean allowPushdownJoinCondition = true;
    MapDPlanner planner =
            getPlanner(allowCorrelatedSubQueryExpansion, allowPushdownJoinCondition);

    SqlNode node = processSQL(sql, parserOptions.isLegacySyntax(), planner);

    if (node.isA(DELETE)) {
      allowCorrelatedSubQueryExpansion = false;
      planner = getPlanner(allowCorrelatedSubQueryExpansion, allowPushdownJoinCondition);
      node = processSQL(sql, parserOptions.isLegacySyntax(), planner);
    } else if (node.isA(UPDATE)) {
      SqlUpdate update = (SqlUpdate) node;
      update = (SqlUpdate) planner.validate(update);
      return rewriteUpdateAsSelect(update, parserOptions);
    }

    if (parserOptions.isLegacySyntax()) {
      // close original planner
      planner.close();
      // create a new one
      planner = getPlanner(allowCorrelatedSubQueryExpansion, allowPushdownJoinCondition);
      node = processSQL(node.toSqlString(SqlDialect.CALCITE).toString(), false, planner);
    }

    boolean is_select_star = isSelectStar(node);

    SqlNode validateR = planner.validate(node);
    SqlSelect validate_select = getSelectChild(validateR);

    // Hide rowid from select * queries
    if (parserOptions.isLegacySyntax() && is_select_star && validate_select != null) {
      SqlNodeList proj_exprs = ((SqlSelect) validateR).getSelectList();
      SqlNodeList new_proj_exprs = new SqlNodeList(proj_exprs.getParserPosition());
      for (SqlNode proj_expr : proj_exprs) {
        final SqlNode unaliased_proj_expr = getUnaliasedExpression(proj_expr);

        if (unaliased_proj_expr instanceof SqlIdentifier) {
          if ((((SqlIdentifier) unaliased_proj_expr).toString().toLowerCase())
                          .endsWith(".rowid")) {
            continue;
          }
        }
        new_proj_exprs.add(proj_expr);
      }
      validate_select.setSelectList(new_proj_exprs);

      // trick planner back into correct state for validate
      planner.close();
      // create a new one
      planner = getPlanner(allowCorrelatedSubQueryExpansion, allowPushdownJoinCondition);
      processSQL(validateR.toSqlString(SqlDialect.CALCITE).toString(), false, planner);
      // now validate the new modified SqlNode;
      validateR = planner.validate(validateR);
    }

    planner.setFilterPushDownInfo(parserOptions.getFilterPushDownInfo());
    RelRoot relR = planner.rel(validateR);
    relR = replaceIsTrue(planner.getTypeFactory(), relR);
    planner.close();

    if (!parserOptions.isViewOptimizeEnabled()) {
      return relR;
    } else {
      // check to see if a view is involved in the query
      boolean foundView = false;
      MapDSchema schema = new MapDSchema(
              dataDir, this, mapdPort, mapdUser, sock_transport_properties);
      SqlIdentifierCapturer capturer =
              captureIdentifiers(sql, parserOptions.isLegacySyntax());
      for (String name : capturer.selects) {
        MapDTable table = (MapDTable) schema.getTable(name);
        if (null == table) {
          throw new RuntimeException("table/view not found: " + name);
        }
        if (table instanceof MapDView) {
          foundView = true;
        }
      }

      if (!foundView) {
        return relR;
      }

      // do some calcite based optimization
      // will allow duplicate projects to merge
      ProjectMergeRule projectMergeRule =
              new ProjectMergeRule(true, RelFactories.LOGICAL_BUILDER);
      final Program program =
              Programs.hep(ImmutableList.of(FilterProjectTransposeRule.INSTANCE,
                                   projectMergeRule,
                                   ProjectProjectRemoveRule.INSTANCE,
                                   FilterMergeRule.INSTANCE,
                                   JoinProjectTransposeRule.LEFT_PROJECT_INCLUDE_OUTER,
                                   JoinProjectTransposeRule.RIGHT_PROJECT_INCLUDE_OUTER,
                                   JoinProjectTransposeRule.BOTH_PROJECT_INCLUDE_OUTER),
                      true,
                      DefaultRelMetadataProvider.INSTANCE);

      RelNode oldRel;
      RelNode newRel = relR.project();

      do {
        oldRel = newRel;
        newRel = program.run(null,
                oldRel,
                null,
                ImmutableList.<RelOptMaterialization>of(),
                ImmutableList.<RelOptLattice>of());
        // there must be a better way to compare these
      } while (!RelOptUtil.toString(oldRel).equals(RelOptUtil.toString(newRel)));
      RelRoot optRel = RelRoot.of(newRel, relR.kind);
      return optRel;
    }
  }

  private RelRoot replaceIsTrue(final RelDataTypeFactory typeFactory, RelRoot root) {
    final RexShuttle callShuttle = new RexShuttle() {
      RexBuilder builder = new RexBuilder(typeFactory);

      public RexNode visitCall(RexCall call) {
        call = (RexCall) super.visitCall(call);
        if (call.getKind() == SqlKind.IS_TRUE) {
          return builder.makeCall(SqlStdOperatorTable.AND,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NOT_NULL, call.getOperands().get(0)),
                  call.getOperands().get(0));
        } else if (call.getKind() == SqlKind.IS_NOT_TRUE) {
          return builder.makeCall(SqlStdOperatorTable.OR,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NULL, call.getOperands().get(0)),
                  builder.makeCall(SqlStdOperatorTable.NOT, call.getOperands().get(0)));
        } else if (call.getKind() == SqlKind.IS_FALSE) {
          return builder.makeCall(SqlStdOperatorTable.AND,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NOT_NULL, call.getOperands().get(0)),
                  builder.makeCall(SqlStdOperatorTable.NOT, call.getOperands().get(0)));
        } else if (call.getKind() == SqlKind.IS_NOT_FALSE) {
          return builder.makeCall(SqlStdOperatorTable.OR,
                  builder.makeCall(
                          SqlStdOperatorTable.IS_NULL, call.getOperands().get(0)),
                  call.getOperands().get(0));
        }

        return call;
      }
    };

    RelNode node = root.rel.accept(new RelShuttleImpl() {
      @Override
      protected RelNode visitChild(RelNode parent, int i, RelNode child) {
        RelNode node = super.visitChild(parent, i, child);
        return node.accept(callShuttle);
      }
    });

    return new RelRoot(
            node, root.validatedRowType, root.kind, root.fields, root.collation);
  }

  private static SqlNode getUnaliasedExpression(final SqlNode node) {
    if (node instanceof SqlBasicCall
            && ((SqlBasicCall) node).getOperator() instanceof SqlAsOperator) {
      SqlNode[] operands = ((SqlBasicCall) node).getOperands();
      return operands[0];
    }
    return node;
  }

  private static boolean isSelectStar(SqlNode node) {
    SqlSelect select_node = getSelectChild(node);
    if (select_node == null) {
      return false;
    }
    SqlNode from = getUnaliasedExpression(select_node.getFrom());
    if (from instanceof SqlCall) {
      return false;
    }
    SqlNodeList proj_exprs = select_node.getSelectList();
    if (proj_exprs.size() != 1) {
      return false;
    }
    SqlNode proj_expr = proj_exprs.get(0);
    if (!(proj_expr instanceof SqlIdentifier)) {
      return false;
    }
    return ((SqlIdentifier) proj_expr).isStar();
  }

  private static SqlSelect getSelectChild(SqlNode node) {
    if (node instanceof SqlSelect) {
      return (SqlSelect) node;
    }
    if (node instanceof SqlOrderBy) {
      SqlOrderBy order_by_node = (SqlOrderBy) node;
      if (order_by_node.query instanceof SqlSelect) {
        return (SqlSelect) order_by_node.query;
      }
    }
    return null;
  }

  private SqlNode processSQL(String sql, final boolean legacy_syntax, Planner planner)
          throws SqlParseException {
    SqlNode parseR = null;
    try {
      parseR = planner.parse(sql);
      MAPDLOGGER.debug(" node is \n" + parseR.toString());
    } catch (SqlParseException ex) {
      MAPDLOGGER.error("failed to process SQL '" + sql + "' \n" + ex.toString());
      throw ex;
    }

    if (!legacy_syntax) {
      return parseR;
    }
    RelDataTypeFactory typeFactory = planner.getTypeFactory();
    SqlSelect select_node = null;
    if (parseR instanceof SqlSelect) {
      select_node = (SqlSelect) parseR;
      desugar(select_node, typeFactory);
    } else if (parseR instanceof SqlOrderBy) {
      SqlOrderBy order_by_node = (SqlOrderBy) parseR;
      if (order_by_node.query instanceof SqlSelect) {
        select_node = (SqlSelect) order_by_node.query;
        SqlOrderBy new_order_by_node = desugar(select_node, order_by_node, typeFactory);
        if (new_order_by_node != null) {
          return new_order_by_node;
        }
      }
    }
    return parseR;
  }

  private void desugar(SqlSelect select_node, RelDataTypeFactory typeFactory) {
    desugar(select_node, null, typeFactory);
  }

  private SqlOrderBy desugar(SqlSelect select_node,
          SqlOrderBy order_by_node,
          RelDataTypeFactory typeFactory) {
    MAPDLOGGER.debug("desugar: before: " + select_node.toString());
    desugarExpression(select_node.getFrom(), typeFactory);
    desugarExpression(select_node.getWhere(), typeFactory);
    SqlNodeList select_list = select_node.getSelectList();
    SqlNodeList new_select_list = new SqlNodeList(select_list.getParserPosition());
    java.util.Map<String, SqlNode> id_to_expr = new java.util.HashMap<String, SqlNode>();
    for (SqlNode proj : select_list) {
      if (!(proj instanceof SqlBasicCall)) {
        new_select_list.add(proj);
        continue;
      }
      SqlBasicCall proj_call = (SqlBasicCall) proj;
      new_select_list.add(expand(proj_call, id_to_expr, typeFactory));
    }
    select_node.setSelectList(new_select_list);
    SqlNodeList group_by_list = select_node.getGroup();
    if (group_by_list != null) {
      select_node.setGroupBy(expand(group_by_list, id_to_expr, typeFactory));
    }
    SqlNode having = select_node.getHaving();
    if (having != null) {
      expand(having, id_to_expr, typeFactory);
    }
    SqlOrderBy new_order_by_node = null;
    if (order_by_node != null && order_by_node.orderList != null
            && order_by_node.orderList.size() > 0) {
      SqlNodeList new_order_by_list =
              expand(order_by_node.orderList, id_to_expr, typeFactory);
      new_order_by_node = new SqlOrderBy(order_by_node.getParserPosition(),
              select_node,
              new_order_by_list,
              order_by_node.offset,
              order_by_node.fetch);
    }

    MAPDLOGGER.debug("desugar:  after: " + select_node.toString());
    return new_order_by_node;
  }

  private void desugarExpression(SqlNode node, RelDataTypeFactory typeFactory) {
    if (node instanceof SqlSelect) {
      desugar((SqlSelect) node, typeFactory);
      return;
    }
    if (!(node instanceof SqlBasicCall)) {
      return;
    }
    SqlBasicCall basic_call = (SqlBasicCall) node;
    for (SqlNode operator : basic_call.getOperands()) {
      if (operator instanceof SqlOrderBy) {
        desugarExpression(((SqlOrderBy) operator).query, typeFactory);
      } else {
        desugarExpression(operator, typeFactory);
      }
    }
  }

  private SqlNode expand(final SqlNode node,
          final java.util.Map<String, SqlNode> id_to_expr,
          RelDataTypeFactory typeFactory) {
    MAPDLOGGER.debug("expand: " + node.toString());
    if (node instanceof SqlBasicCall) {
      SqlBasicCall node_call = (SqlBasicCall) node;
      SqlNode[] operands = node_call.getOperands();
      for (int i = 0; i < operands.length; ++i) {
        node_call.setOperand(i, expand(operands[i], id_to_expr, typeFactory));
      }
      SqlNode expanded_variance = expandVariance(node_call, typeFactory);
      if (expanded_variance != null) {
        return expanded_variance;
      }
      SqlNode expanded_covariance = expandCovariance(node_call, typeFactory);
      if (expanded_covariance != null) {
        return expanded_covariance;
      }
      SqlNode expanded_correlation = expandCorrelation(node_call, typeFactory);
      if (expanded_correlation != null) {
        return expanded_correlation;
      }
    }
    if (node instanceof SqlSelect) {
      SqlSelect select_node = (SqlSelect) node;
      desugar(select_node, typeFactory);
    }
    return node;
  }

  private SqlNodeList expand(final SqlNodeList group_by_list,
          final java.util.Map<String, SqlNode> id_to_expr,
          RelDataTypeFactory typeFactory) {
    SqlNodeList new_group_by_list = new SqlNodeList(new SqlParserPos(-1, -1));
    for (SqlNode group_by : group_by_list) {
      if (!(group_by instanceof SqlIdentifier)) {
        new_group_by_list.add(expand(group_by, id_to_expr, typeFactory));
        continue;
      }
      SqlIdentifier group_by_id = ((SqlIdentifier) group_by);
      if (id_to_expr.containsKey(group_by_id.toString())) {
        new_group_by_list.add(id_to_expr.get(group_by_id.toString()));
      } else {
        new_group_by_list.add(group_by);
      }
    }
    return new_group_by_list;
  }

  private SqlNode expandVariance(
          final SqlBasicCall proj_call, RelDataTypeFactory typeFactory) {
    // Expand variance aggregates that are not supported natively
    if (proj_call.operandCount() != 1) {
      return null;
    }
    boolean biased;
    boolean sqrt;
    boolean flt;
    if (proj_call.getOperator().isName("STDDEV_POP", false)) {
      biased = true;
      sqrt = true;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("STDDEV_POP_FLOAT")) {
      biased = true;
      sqrt = true;
      flt = true;
    } else if (proj_call.getOperator().isName("STDDEV_SAMP", false)
            || proj_call.getOperator().getName().equalsIgnoreCase("STDDEV")) {
      biased = false;
      sqrt = true;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("STDDEV_SAMP_FLOAT")
            || proj_call.getOperator().getName().equalsIgnoreCase("STDDEV_FLOAT")) {
      biased = false;
      sqrt = true;
      flt = true;
    } else if (proj_call.getOperator().isName("VAR_POP", false)) {
      biased = true;
      sqrt = false;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("VAR_POP_FLOAT")) {
      biased = true;
      sqrt = false;
      flt = true;
    } else if (proj_call.getOperator().isName("VAR_SAMP", false)
            || proj_call.getOperator().getName().equalsIgnoreCase("VARIANCE")) {
      biased = false;
      sqrt = false;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("VAR_SAMP_FLOAT")
            || proj_call.getOperator().getName().equalsIgnoreCase("VARIANCE_FLOAT")) {
      biased = false;
      sqrt = false;
      flt = true;
    } else {
      return null;
    }
    final SqlNode operand = proj_call.operand(0);
    final SqlParserPos pos = proj_call.getParserPosition();
    SqlNode expanded_proj_call =
            expandVariance(pos, operand, biased, sqrt, flt, typeFactory);
    MAPDLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
    MAPDLOGGER.debug("to : " + expanded_proj_call.toString());
    return expanded_proj_call;
  }

  private SqlNode expandVariance(final SqlParserPos pos,
          final SqlNode operand,
          boolean biased,
          boolean sqrt,
          boolean flt,
          RelDataTypeFactory typeFactory) {
    // stddev_pop(x) ==>
    // power(
    // (sum(x * x) - sum(x) * sum(x) / (case count(x) when 0 then NULL else count(x)
    // end)) / (case count(x) when 0 then NULL else count(x) end), .5)
    //
    // stddev_samp(x) ==>
    // power(
    // (sum(x * x) - sum(x) * sum(x) / (case count(x) when 0 then NULL else count(x)
    // )) / ((case count(x) when 1 then NULL else count(x) - 1 end)), .5)
    //
    // var_pop(x) ==>
    // (sum(x * x) - sum(x) * sum(x) / ((case count(x) when 0 then NULL else
    // count(x)
    // end))) / ((case count(x) when 0 then NULL else count(x) end))
    //
    // var_samp(x) ==>
    // (sum(x * x) - sum(x) * sum(x) / ((case count(x) when 0 then NULL else
    // count(x)
    // end))) / ((case count(x) when 1 then NULL else count(x) - 1 end))
    //
    final SqlNode arg = SqlStdOperatorTable.CAST.createCall(pos,
            operand,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    final SqlNode argSquared = SqlStdOperatorTable.MULTIPLY.createCall(pos, arg, arg);
    final SqlNode sumArgSquared = SqlStdOperatorTable.SUM.createCall(pos, argSquared);
    final SqlNode sum = SqlStdOperatorTable.SUM.createCall(pos, arg);
    final SqlNode sumSquared = SqlStdOperatorTable.MULTIPLY.createCall(pos, sum, sum);
    final SqlNode count = SqlStdOperatorTable.COUNT.createCall(pos, arg);
    final SqlLiteral nul = SqlLiteral.createNull(pos);
    final SqlNumericLiteral zero = SqlLiteral.createExactNumeric("0", pos);
    final SqlNode countEqZero = SqlStdOperatorTable.EQUALS.createCall(pos, count, zero);
    SqlNodeList whenList = new SqlNodeList(pos);
    SqlNodeList thenList = new SqlNodeList(pos);
    whenList.add(countEqZero);
    thenList.add(nul);
    final SqlNode int_denominator = SqlStdOperatorTable.CASE.createCall(
            null, pos, null, whenList, thenList, count);
    final SqlNode denominator = SqlStdOperatorTable.CAST.createCall(pos,
            int_denominator,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    final SqlNode avgSumSquared =
            SqlStdOperatorTable.DIVIDE.createCall(pos, sumSquared, denominator);
    final SqlNode diff =
            SqlStdOperatorTable.MINUS.createCall(pos, sumArgSquared, avgSumSquared);
    final SqlNode denominator1;
    if (biased) {
      denominator1 = denominator;
    } else {
      final SqlNumericLiteral one = SqlLiteral.createExactNumeric("1", pos);
      final SqlNode countEqOne = SqlStdOperatorTable.EQUALS.createCall(pos, count, one);
      final SqlNode countMinusOne = SqlStdOperatorTable.MINUS.createCall(pos, count, one);
      SqlNodeList whenList1 = new SqlNodeList(pos);
      SqlNodeList thenList1 = new SqlNodeList(pos);
      whenList1.add(countEqOne);
      thenList1.add(nul);
      final SqlNode int_denominator1 = SqlStdOperatorTable.CASE.createCall(
              null, pos, null, whenList1, thenList1, countMinusOne);
      denominator1 = SqlStdOperatorTable.CAST.createCall(pos,
              int_denominator1,
              SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                      flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    }
    final SqlNode div = SqlStdOperatorTable.DIVIDE.createCall(pos, diff, denominator1);
    SqlNode result = div;
    if (sqrt) {
      final SqlNumericLiteral half = SqlLiteral.createExactNumeric("0.5", pos);
      result = SqlStdOperatorTable.POWER.createCall(pos, div, half);
    }
    return SqlStdOperatorTable.CAST.createCall(pos,
            result,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
  }

  private SqlNode expandCovariance(
          final SqlBasicCall proj_call, RelDataTypeFactory typeFactory) {
    // Expand covariance aggregates
    if (proj_call.operandCount() != 2) {
      return null;
    }
    boolean pop;
    boolean flt;
    if (proj_call.getOperator().isName("COVAR_POP", false)) {
      pop = true;
      flt = false;
    } else if (proj_call.getOperator().isName("COVAR_SAMP", false)) {
      pop = false;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("COVAR_POP_FLOAT")) {
      pop = true;
      flt = true;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("COVAR_SAMP_FLOAT")) {
      pop = false;
      flt = true;
    } else {
      return null;
    }
    final SqlNode operand0 = proj_call.operand(0);
    final SqlNode operand1 = proj_call.operand(1);
    final SqlParserPos pos = proj_call.getParserPosition();
    SqlNode expanded_proj_call =
            expandCovariance(pos, operand0, operand1, pop, flt, typeFactory);
    MAPDLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
    MAPDLOGGER.debug("to : " + expanded_proj_call.toString());
    return expanded_proj_call;
  }

  private SqlNode expandCovariance(SqlParserPos pos,
          final SqlNode operand0,
          final SqlNode operand1,
          boolean pop,
          boolean flt,
          RelDataTypeFactory typeFactory) {
    // covar_pop(x, y) ==> avg(x * y) - avg(x) * avg(y)
    // covar_samp(x, y) ==> (sum(x * y) - sum(x) * avg(y))
    // ((case count(x) when 1 then NULL else count(x) - 1 end))
    final SqlNode arg0 = SqlStdOperatorTable.CAST.createCall(operand0.getParserPosition(),
            operand0,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    final SqlNode arg1 = SqlStdOperatorTable.CAST.createCall(operand1.getParserPosition(),
            operand1,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    final SqlNode mulArg = SqlStdOperatorTable.MULTIPLY.createCall(pos, arg0, arg1);
    final SqlNode avgArg1 = SqlStdOperatorTable.AVG.createCall(pos, arg1);
    if (pop) {
      final SqlNode avgMulArg = SqlStdOperatorTable.AVG.createCall(pos, mulArg);
      final SqlNode avgArg0 = SqlStdOperatorTable.AVG.createCall(pos, arg0);
      final SqlNode mulAvgAvg =
              SqlStdOperatorTable.MULTIPLY.createCall(pos, avgArg0, avgArg1);
      final SqlNode covarPop =
              SqlStdOperatorTable.MINUS.createCall(pos, avgMulArg, mulAvgAvg);
      return SqlStdOperatorTable.CAST.createCall(pos,
              covarPop,
              SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                      flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    }
    final SqlNode sumMulArg = SqlStdOperatorTable.SUM.createCall(pos, mulArg);
    final SqlNode sumArg0 = SqlStdOperatorTable.SUM.createCall(pos, arg0);
    final SqlNode mulSumAvg =
            SqlStdOperatorTable.MULTIPLY.createCall(pos, sumArg0, avgArg1);
    final SqlNode sub = SqlStdOperatorTable.MINUS.createCall(pos, sumMulArg, mulSumAvg);
    final SqlNode count = SqlStdOperatorTable.COUNT.createCall(pos, operand0);
    final SqlNumericLiteral one = SqlLiteral.createExactNumeric("1", pos);
    final SqlNode countEqOne = SqlStdOperatorTable.EQUALS.createCall(pos, count, one);
    final SqlNode countMinusOne = SqlStdOperatorTable.MINUS.createCall(pos, count, one);
    final SqlLiteral nul = SqlLiteral.createNull(pos);
    SqlNodeList whenList1 = new SqlNodeList(pos);
    SqlNodeList thenList1 = new SqlNodeList(pos);
    whenList1.add(countEqOne);
    thenList1.add(nul);
    final SqlNode int_denominator = SqlStdOperatorTable.CASE.createCall(
            null, pos, null, whenList1, thenList1, countMinusOne);
    final SqlNode denominator = SqlStdOperatorTable.CAST.createCall(pos,
            int_denominator,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
    final SqlNode covarSamp =
            SqlStdOperatorTable.DIVIDE.createCall(pos, sub, denominator);
    return SqlStdOperatorTable.CAST.createCall(pos,
            covarSamp,
            SqlTypeUtil.convertTypeToSpec(typeFactory.createSqlType(
                    flt ? SqlTypeName.FLOAT : SqlTypeName.DOUBLE)));
  }

  private SqlNode expandCorrelation(
          final SqlBasicCall proj_call, RelDataTypeFactory typeFactory) {
    // Expand correlation coefficient
    if (proj_call.operandCount() != 2) {
      return null;
    }
    boolean flt;
    if (proj_call.getOperator().isName("CORR", false)
            || proj_call.getOperator().getName().equalsIgnoreCase("CORRELATION")) {
      // expand correlation coefficient
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("CORR_FLOAT")
            || proj_call.getOperator().getName().equalsIgnoreCase("CORRELATION_FLOAT")) {
      // expand correlation coefficient
      flt = true;
    } else {
      return null;
    }
    // corr(x, y) ==> (avg(x * y) - avg(x) * avg(y)) / (stddev_pop(x) *
    // stddev_pop(y))
    // ==> covar_pop(x, y) / (stddev_pop(x) * stddev_pop(y))
    final SqlNode operand0 = proj_call.operand(0);
    final SqlNode operand1 = proj_call.operand(1);
    final SqlParserPos pos = proj_call.getParserPosition();
    SqlNode covariance =
            expandCovariance(pos, operand0, operand1, true, flt, typeFactory);
    SqlNode stddev0 = expandVariance(pos, operand0, true, true, flt, typeFactory);
    SqlNode stddev1 = expandVariance(pos, operand1, true, true, flt, typeFactory);
    final SqlNode mulStddev =
            SqlStdOperatorTable.MULTIPLY.createCall(pos, stddev0, stddev1);
    final SqlNumericLiteral zero = SqlLiteral.createExactNumeric("0.0", pos);
    final SqlNode mulStddevEqZero =
            SqlStdOperatorTable.EQUALS.createCall(pos, mulStddev, zero);
    final SqlLiteral nul = SqlLiteral.createNull(pos);
    SqlNodeList whenList1 = new SqlNodeList(pos);
    SqlNodeList thenList1 = new SqlNodeList(pos);
    whenList1.add(mulStddevEqZero);
    thenList1.add(nul);
    final SqlNode denominator = SqlStdOperatorTable.CASE.createCall(
            null, pos, null, whenList1, thenList1, mulStddev);
    final SqlNode expanded_proj_call =
            SqlStdOperatorTable.DIVIDE.createCall(pos, covariance, denominator);
    MAPDLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
    MAPDLOGGER.debug("to : " + expanded_proj_call.toString());
    return expanded_proj_call;
  }

  /**
   * Creates an operator table.
   *
   * @param extSigs
   * @return New operator table
   */
  protected SqlOperatorTable createOperatorTable(
          final Map<String, ExtensionFunction> extSigs) {
    final MapDSqlOperatorTable tempOpTab =
            new MapDSqlOperatorTable(SqlStdOperatorTable.instance());
    // MAT 11 Nov 2015
    // Example of how to add custom function
    MapDSqlOperatorTable.addUDF(tempOpTab, extSigs);
    return tempOpTab;
  }

  public SqlIdentifierCapturer captureIdentifiers(String sql, boolean legacy_syntax)
          throws SqlParseException {
    try {
      Planner planner = getPlanner();
      SqlNode node = processSQL(sql, legacy_syntax, planner);
      SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
      capturer.scan(node);
      return capturer;
    } catch (Exception | Error e) {
      MAPDLOGGER.error("Error parsing sql: " + sql, e);
      return new SqlIdentifierCapturer();
    }
  }

  public int getCallCount() {
    return callCount;
  }

  public void updateMetaData(String schema, String table) {
    MAPDLOGGER.debug("schema :" + schema + " table :" + table);
    MapDSchema mapd =
            new MapDSchema(dataDir, this, mapdPort, null, sock_transport_properties);
    mapd.updateMetaData(schema, table);
  }

  protected RelDataTypeSystem createTypeSystem() {
    final MapDTypeSystem typeSystem = new MapDTypeSystem();
    return typeSystem;
  }
}

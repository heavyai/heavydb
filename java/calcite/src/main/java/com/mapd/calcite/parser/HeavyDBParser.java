/*
 * Copyright 2022 HEAVY.AI, Inc.
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
import com.mapd.calcite.rel.rules.FilterTableFunctionMultiInputTransposeRule;
import com.mapd.common.SockTransportProperties;
import com.mapd.metadata.MetaConnect;
import com.mapd.parser.extension.ddl.ExtendedSqlParser;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;
import com.mapd.parser.hint.HeavyDBHintStrategyTable;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.config.CalciteConnectionConfigImpl;
import org.apache.calcite.config.CalciteConnectionProperty;
import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.prepare.HeavyDBPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.RelShuttleImpl;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.core.TableModify.Operation;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalTableModify;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.Restriction;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.*;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.advise.SqlAdvisorValidator;
import org.apache.calcite.sql.dialect.CalciteSqlDialect;
import org.apache.calcite.sql.fun.SqlCase;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.util.SqlBasicVisitor;
import org.apache.calcite.sql.util.SqlShuttle;
import org.apache.calcite.sql.util.SqlVisitor;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.*;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiPredicate;
import java.util.function.Supplier;
import java.util.stream.Stream;

import ai.heavy.thrift.server.TColumnType;
import ai.heavy.thrift.server.TDatumType;
import ai.heavy.thrift.server.TEncodingType;
import ai.heavy.thrift.server.TTableDetails;

public final class HeavyDBParser {
  public static final ThreadLocal<HeavyDBParser> CURRENT_PARSER = new ThreadLocal<>();
  private static final EnumSet<SqlKind> SCALAR =
          EnumSet.of(SqlKind.SCALAR_QUERY, SqlKind.SELECT);
  private static final EnumSet<SqlKind> EXISTS = EnumSet.of(SqlKind.EXISTS);
  private static final EnumSet<SqlKind> DELETE = EnumSet.of(SqlKind.DELETE);
  private static final EnumSet<SqlKind> UPDATE = EnumSet.of(SqlKind.UPDATE);
  private static final EnumSet<SqlKind> IN = EnumSet.of(SqlKind.IN);
  private static final EnumSet<SqlKind> ARRAY_VALUE =
          EnumSet.of(SqlKind.ARRAY_VALUE_CONSTRUCTOR);
  private static final EnumSet<SqlKind> OTHER_FUNCTION =
          EnumSet.of(SqlKind.OTHER_FUNCTION);

  final static Logger HEAVYDBLOGGER = LoggerFactory.getLogger(HeavyDBParser.class);

  private final Supplier<HeavyDBSqlOperatorTable> dbSqlOperatorTable;
  private final String dataDir;

  private int callCount = 0;
  private final int dbPort;
  private HeavyDBUser dbUser;
  private SockTransportProperties sock_transport_properties = null;

  private static Map<String, Boolean> SubqueryCorrMemo = new ConcurrentHashMap<>();

  public HeavyDBParser(String dataDir,
          final Supplier<HeavyDBSqlOperatorTable> dbSqlOperatorTable,
          int dbPort,
          SockTransportProperties skT) {
    this.dataDir = dataDir;
    this.dbSqlOperatorTable = dbSqlOperatorTable;
    this.dbPort = dbPort;
    this.sock_transport_properties = skT;
  }

  public void clearMemo() {
    SubqueryCorrMemo.clear();
  }

  private static final Context DB_CONNECTION_CONTEXT = new Context() {
    HeavyDBTypeSystem myTypeSystem = new HeavyDBTypeSystem();
    CalciteConnectionConfig config = new CalciteConnectionConfigImpl(new Properties()) {
      {
        properties.put(CalciteConnectionProperty.CASE_SENSITIVE.camelName(),
                String.valueOf(false));
        properties.put(CalciteConnectionProperty.CONFORMANCE.camelName(),
                String.valueOf(SqlConformanceEnum.LENIENT));
      }

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

  private HeavyDBPlanner getPlanner() {
    return getPlanner(true, false, false);
  }

  private boolean isCorrelated(SqlNode expression) {
    String queryString = expression.toSqlString(CalciteSqlDialect.DEFAULT).getSql();
    Boolean isCorrelatedSubquery = SubqueryCorrMemo.get(queryString);
    if (null != isCorrelatedSubquery) {
      return isCorrelatedSubquery;
    }

    try {
      HeavyDBParser parser = new HeavyDBParser(
              dataDir, dbSqlOperatorTable, dbPort, sock_transport_properties);
      HeavyDBParserOptions options = new HeavyDBParserOptions();
      parser.setUser(dbUser);
      parser.processSql(expression, options);
    } catch (Exception e) {
      // if we are not able to parse, then assume correlated
      SubqueryCorrMemo.put(queryString, true);
      return true;
    }
    SubqueryCorrMemo.put(queryString, false);
    return false;
  }

  private boolean isHashJoinableType(TColumnType type) {
    switch (type.getCol_type().type) {
      case TINYINT:
      case SMALLINT:
      case INT:
      case BIGINT: {
        return true;
      }
      case STR: {
        return type.col_type.encoding == TEncodingType.DICT;
      }
      default: {
        return false;
      }
    }
  }

  private boolean isColumnHashJoinable(
          List<String> joinColumnIdentifier, MetaConnect mc) {
    try {
      TTableDetails tableDetails = mc.get_table_details(joinColumnIdentifier.get(0));
      return null
              != tableDetails.row_desc.stream()
                         .filter(c
                                 -> c.col_name.toLowerCase(Locale.ROOT)
                                                 .equals(joinColumnIdentifier.get(1)
                                                                 .toLowerCase(
                                                                         Locale.ROOT))
                                         && isHashJoinableType(c))
                         .findFirst()
                         .orElse(null);
    } catch (Exception e) {
      return false;
    }
  }

  private HeavyDBPlanner getPlanner(final boolean allowSubQueryExpansion,
          final boolean isWatchdogEnabled,
          final boolean isDistributedMode) {
    HeavyDBUser user = new HeavyDBUser(dbUser.getUser(),
            dbUser.getSession(),
            dbUser.getDB(),
            -1,
            ImmutableList.of());
    final MetaConnect mc =
            new MetaConnect(dbPort, dataDir, user, this, sock_transport_properties);
    BiPredicate<SqlNode, SqlNode> expandPredicate = new BiPredicate<SqlNode, SqlNode>() {
      @Override
      public boolean test(SqlNode root, SqlNode expression) {
        if (!allowSubQueryExpansion) {
          return false;
        }

        if (expression.isA(EXISTS) || expression.isA(IN)) {
          // try to expand subquery by EXISTS and IN clauses by default
          // note that current Calcite decorrelator fails to flat
          // NOT-IN clause in some cases, so we do not decorrelate it for now

          if (expression.isA(IN)) {
            // If we enable watchdog, we suffer from large projection exception in many
            // cases since decorrelation needs de-duplication step which adds project -
            // aggregate logic. And the added project is the source of the exception when
            // its underlying table is large. Thus, we enable IN-clause decorrelation
            // under watchdog iff we explicitly have correlated join in IN-clause
            if (expression instanceof SqlCall) {
              SqlCall outerSelectCall = (SqlCall) expression;
              if (outerSelectCall.getOperandList().size() == 2) {
                // if IN clause is correlated, its second operand of corresponding
                // expression is SELECT clause which indicates a correlated subquery.
                // Here, an expression "f.val IN (SELECT ...)" has two operands.
                // Since we have interest in its subquery, so try to check whether
                // the second operand, i.e., call.getOperandList().get(1)
                // is a type of SqlSelect and also is correlated.
                if (outerSelectCall.getOperandList().get(1) instanceof SqlSelect) {
                  // the below checking logic is to allow IN-clause decorrelation
                  // if it has hash joinable IN expression without correlated join
                  // i.e., SELECT ... WHERE a.intVal IN (SELECT b.intVal FROM b) ...;
                  SqlSelect innerSelectCall =
                          (SqlSelect) outerSelectCall.getOperandList().get(1);
                  if (innerSelectCall.hasWhere()) {
                    // IN-clause may have correlated join within subquery's WHERE clause
                    // i.e., f.val IN (SELECT r.val FROM R r WHERE f.val2 = r.val2)
                    // then we have to deccorrelate the IN-clause
                    JoinOperatorChecker joinOperatorChecker = new JoinOperatorChecker();
                    if (joinOperatorChecker.containsExpression(
                                innerSelectCall.getWhere())) {
                      return true;
                    }
                  }
                  if (isDistributedMode) {
                    // we temporarily disable IN-clause decorrelation in dist mode
                    // todo (yoonmin) : relax this in dist mode when available
                    return false;
                  }
                  boolean hasHashJoinableExpression = false;
                  if (isWatchdogEnabled) {
                    // when watchdog is enabled, we try to selectively allow decorrelation
                    // iff IN-expression is between two columns that both are hash
                    // joinable
                    Map<String, String> tableAliasMap = new HashMap<>();
                    if (root instanceof SqlSelect) {
                      tableAliasFinder(((SqlSelect) root).getFrom(), tableAliasMap);
                    }
                    tableAliasFinder(innerSelectCall.getFrom(), tableAliasMap);
                    if (outerSelectCall.getOperandList().get(0) instanceof SqlIdentifier
                            && innerSelectCall.getSelectList().get(0)
                                            instanceof SqlIdentifier) {
                      SqlIdentifier outerColIdentifier =
                              (SqlIdentifier) outerSelectCall.getOperandList().get(0);
                      SqlIdentifier innerColIdentifier =
                              (SqlIdentifier) innerSelectCall.getSelectList().get(0);
                      if (tableAliasMap.containsKey(outerColIdentifier.names.get(0))
                              && tableAliasMap.containsKey(
                                      innerColIdentifier.names.get(0))) {
                        String outerTableName =
                                tableAliasMap.get(outerColIdentifier.names.get(0));
                        String innerTableName =
                                tableAliasMap.get(innerColIdentifier.names.get(0));
                        if (isColumnHashJoinable(ImmutableList.of(outerTableName,
                                                         outerColIdentifier.names.get(1)),
                                    mc)
                                && isColumnHashJoinable(
                                        ImmutableList.of(innerTableName,
                                                innerColIdentifier.names.get(1)),
                                        mc)) {
                          hasHashJoinableExpression = true;
                        }
                      }
                    }
                    if (!hasHashJoinableExpression) {
                      return false;
                    }
                  }
                }
              }
            }
            if (root instanceof SqlSelect) {
              SqlSelect selectCall = (SqlSelect) root;
              if (new ExpressionListedInSelectClauseChecker().containsExpression(
                          selectCall, expression)) {
                // occasionally, Calcite cannot properly decorrelate IN-clause listed in
                // SELECT clause e.g., SELECT x, CASE WHEN x in (SELECT x FROM R) ... FROM
                // ... in that case we disable input query's decorrelation
                return false;
              }
              if (null != selectCall.getWhere()) {
                if (new ExpressionListedAsChildOROperatorChecker().containsExpression(
                            selectCall.getWhere(), expression)) {
                  // Decorrelation logic of the current Calcite cannot cover IN-clause
                  // well if it is listed as a child operand of OR-op
                  return false;
                }
              }
              if (null != selectCall.getHaving()) {
                if (new ExpressionListedAsChildOROperatorChecker().containsExpression(
                            selectCall.getHaving(), expression)) {
                  // Decorrelation logic of the current Calcite cannot cover IN-clause
                  // well if it is listed as a child operand of OR-op
                  return false;
                }
              }
            }
          }

          // otherwise, let's decorrelate the expression
          return true;
        }

        // special handling of sub-queries
        if (expression.isA(SCALAR) && isCorrelated(expression)) {
          // only expand if it is correlated.
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

        // per default we do not want to expand
        return false;
      }
    };

    final HeavyDBSchema defaultSchema = new HeavyDBSchema(
            dataDir, this, dbPort, dbUser, sock_transport_properties, dbUser.getDB());
    final SchemaPlus rootSchema = Frameworks.createRootSchema(true);
    final SchemaPlus defaultSchemaPlus = rootSchema.add(dbUser.getDB(), defaultSchema);
    for (String db : mc.getDatabases()) {
      if (!db.equalsIgnoreCase(dbUser.getDB())) {
        rootSchema.add(db,
                new HeavyDBSchema(
                        dataDir, this, dbPort, dbUser, sock_transport_properties, db));
      }
    }

    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(defaultSchemaPlus)
                    .operatorTable(dbSqlOperatorTable.get())
                    .parserConfig(SqlParser.configBuilder()
                                          .setConformance(SqlConformanceEnum.LENIENT)
                                          .setUnquotedCasing(Casing.UNCHANGED)
                                          .setCaseSensitive(false)
                                          // allow identifiers of up to 512 chars
                                          .setIdentifierMaxLength(512)
                                          .setParserFactory(ExtendedSqlParser.FACTORY)
                                          .build())
                    .sqlToRelConverterConfig(
                            SqlToRelConverter
                                    .configBuilder()
                                    // enable sub-query expansion (de-correlation)
                                    .withExpandPredicate(expandPredicate)
                                    // allow as many as possible IN operator values
                                    .withInSubQueryThreshold(Integer.MAX_VALUE)
                                    .withHintStrategyTable(
                                            HeavyDBHintStrategyTable.HINT_STRATEGY_TABLE)
                                    .build())

                    .typeSystem(createTypeSystem())
                    .context(DB_CONNECTION_CONTEXT)
                    .build();
    HeavyDBPlanner planner = new HeavyDBPlanner(config);
    planner.setRestrictions(dbUser.getRestrictions());
    return planner;
  }

  public void setUser(HeavyDBUser dbUser) {
    this.dbUser = dbUser;
  }

  public Pair<String, SqlIdentifierCapturer> process(
          String sql, final HeavyDBParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    final HeavyDBPlanner planner = getPlanner(
            true, parserOptions.isWatchdogEnabled(), parserOptions.isDistributedMode());
    final SqlNode sqlNode = parseSql(sql, parserOptions.isLegacySyntax(), planner);
    String res = processSql(sqlNode, parserOptions);
    SqlIdentifierCapturer capture = captureIdentifiers(sqlNode);
    return new Pair<String, SqlIdentifierCapturer>(res, capture);
  }

  public String buildRATreeAndPerformQueryOptimization(
          String query, final HeavyDBParserOptions parserOptions) throws IOException {
    HeavyDBSchema schema = new HeavyDBSchema(
            dataDir, this, dbPort, dbUser, sock_transport_properties, dbUser.getDB());
    HeavyDBPlanner planner = getPlanner(
            true, parserOptions.isWatchdogEnabled(), parserOptions.isDistributedMode());

    planner.setFilterPushDownInfo(parserOptions.getFilterPushDownInfo());
    RelRoot optRel = planner.buildRATreeAndPerformQueryOptimization(query, schema);
    optRel = replaceIsTrue(planner.getTypeFactory(), optRel);
    return HeavyDBSerializer.toString(optRel.project());
  }

  public String processSql(String sql, final HeavyDBParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    callCount++;

    final HeavyDBPlanner planner = getPlanner(
            true, parserOptions.isWatchdogEnabled(), parserOptions.isDistributedMode());
    final SqlNode sqlNode = parseSql(sql, parserOptions.isLegacySyntax(), planner);

    return processSql(sqlNode, parserOptions);
  }

  public String processSql(
          final SqlNode sqlNode, final HeavyDBParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    callCount++;

    if (sqlNode instanceof JsonSerializableDdl) {
      return ((JsonSerializableDdl) sqlNode).toJsonString();
    }

    if (sqlNode instanceof SqlDdl) {
      return sqlNode.toString();
    }

    final HeavyDBPlanner planner = getPlanner(
            true, parserOptions.isWatchdogEnabled(), parserOptions.isDistributedMode());
    planner.advanceToValidate();

    final RelRoot sqlRel = convertSqlToRelNode(sqlNode, planner, parserOptions);
    RelNode project = sqlRel.project();

    if (parserOptions.isExplain()) {
      return RelOptUtil.toString(sqlRel.project());
    }

    String res = HeavyDBSerializer.toString(project);

    return res;
  }

  public HeavyDBPlanner.CompletionResult getCompletionHints(
          String sql, int cursor, List<String> visible_tables) {
    return getPlanner().getCompletionHints(sql, cursor, visible_tables);
  }

  public HashSet<ImmutableList<String>> resolveSelectIdentifiers(
          SqlIdentifierCapturer capturer) {
    HashSet<ImmutableList<String>> resolved = new HashSet<ImmutableList<String>>();

    for (ImmutableList<String> names : capturer.selects) {
      HeavyDBSchema schema = new HeavyDBSchema(
              dataDir, this, dbPort, dbUser, sock_transport_properties, names.get(1));
      HeavyDBTable table = (HeavyDBTable) schema.getTable(names.get(0));
      if (null == table) {
        throw new RuntimeException("table/view not found: " + names.get(0));
      }

      if (table instanceof HeavyDBView) {
        HeavyDBView view = (HeavyDBView) table;
        resolved.addAll(resolveSelectIdentifiers(view.getAccessedObjects()));
      } else {
        resolved.add(names);
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
      if (select0.isA(EnumSet.of(SqlKind.AS))) {
        select0 = ((SqlCall) select0).getOperandList().get(0);
      }
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

    SqlSelect select = new SqlSelect(ZERO,
            null,
            selectList,
            join,
            where,
            groupBy,
            null,
            null,
            null,
            null,
            null,
            null);
    return select;
  }

  private LogicalTableModify getDummyUpdate(SqlUpdate update)
          throws SqlParseException, ValidationException, RelConversionException {
    SqlIdentifier targetTable = (SqlIdentifier) update.getTargetTable();
    String targetTableName = targetTable.toString();
    HeavyDBPlanner planner = getPlanner();
    String dummySql = "DELETE FROM " + targetTableName;
    SqlNode dummyNode = planner.parse(dummySql);
    dummyNode = planner.validate(dummyNode);
    RelRoot dummyRoot = planner.rel(dummyNode);
    LogicalTableModify dummyModify = (LogicalTableModify) dummyRoot.rel;
    return dummyModify;
  }

  private RelRoot rewriteUpdateAsSelect(
          SqlUpdate update, HeavyDBParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    int correlatedQueriesCount[] = new int[1];
    SqlBasicVisitor<Void> correlatedQueriesCounter = new SqlBasicVisitor<Void>() {
      @Override
      public Void visit(SqlCall call) {
        if (call.isA(SCALAR)
                && ((call instanceof SqlBasicCall && call.operandCount() == 1
                            && !call.operand(0).isA(SCALAR))
                        || !(call instanceof SqlBasicCall))) {
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
              "table modifications with multiple correlated sub-queries not supported.",
              null);
    }

    boolean allowSubqueryDecorrelation = true;
    SqlNode updateCondition = update.getCondition();
    if (null != updateCondition) {
      boolean hasInClause =
              new FindSqlOperator().containsSqlOperator(updateCondition, SqlKind.IN);
      if (hasInClause) {
        SqlNode updateTargetTable = update.getTargetTable();
        if (null != updateTargetTable && updateTargetTable instanceof SqlIdentifier) {
          SqlIdentifier targetTable = (SqlIdentifier) updateTargetTable;
          if (targetTable.names.size() == 2) {
            final MetaConnect mc = new MetaConnect(dbPort,
                    dataDir,
                    dbUser,
                    this,
                    sock_transport_properties,
                    targetTable.names.get(0));
            TTableDetails updateTargetTableDetails =
                    mc.get_table_details(targetTable.names.get(1));
            if (null != updateTargetTableDetails
                    && updateTargetTableDetails.is_temporary) {
              allowSubqueryDecorrelation = false;
            }
          }
        }
      }
    }

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

        if (!(targetColumn instanceof SqlIdentifier)) {
          throw new RuntimeException("Unknown identifier type!");
        }
        SqlIdentifier id = (SqlIdentifier) targetColumn;
        RelDataType fieldType =
                targetTableType.getField(id.names.get(id.names.size() - 1), false, false)
                        .getType();

        if (expression.isA(ARRAY_VALUE) && null != fieldType.getComponentType()) {
          // apply a cast to all array value elements

          SqlDataTypeSpec elementType = new SqlDataTypeSpec(
                  new SqlBasicTypeNameSpec(fieldType.getComponentType().getSqlTypeName(),
                          fieldType.getPrecision(),
                          fieldType.getScale(),
                          null == fieldType.getCharset() ? null
                                                         : fieldType.getCharset().name(),
                          SqlParserPos.ZERO),
                  SqlParserPos.ZERO);
          SqlCall array_expression = (SqlCall) expression;
          ArrayList<SqlNode> values = new ArrayList<>();

          for (SqlNode value : array_expression.getOperandList()) {
            if (value.isA(EnumSet.of(SqlKind.LITERAL))) {
              SqlNode casted_value = new SqlBasicCall(SqlStdOperatorTable.CAST,
                      new SqlNode[] {value, elementType},
                      value.getParserPosition());
              values.add(casted_value);
            } else {
              values.add(value);
            }
          }

          expression = new SqlBasicCall(HeavyDBSqlOperatorTable.ARRAY_VALUE_CONSTRUCTOR,
                  values.toArray(new SqlNode[0]),
                  expression.getParserPosition());
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
              null,
              null);
    }

    HeavyDBPlanner planner = getPlanner(allowSubqueryDecorrelation,
            parserOptions.isWatchdogEnabled(),
            parserOptions.isDistributedMode());
    SqlNode node = null;
    try {
      node = planner.parse(select.toSqlString(CalciteSqlDialect.DEFAULT).getSql());
      node = planner.validate(node);
    } catch (Exception e) {
      HEAVYDBLOGGER.error("Error processing UPDATE rewrite, rewritten stmt was: "
              + select.toSqlString(CalciteSqlDialect.DEFAULT).getSql());
      throw e;
    }

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

    // The magical number here when processing the projection
    // is skipping the OFFSET_IN_FRAGMENT() expression used by
    // update and delete
    int idx = 0;
    for (RexNode exp : project.getProjects()) {
      if (applyRexCast && idx + 1 < project.getProjects().size()) {
        RelDataType expectedFieldType =
                targetTableType.getField(fields.get(idx), false, false).getType();
        boolean is_array_kind = exp.isA(ARRAY_VALUE);
        boolean is_func_kind = exp.isA(OTHER_FUNCTION);
        // runtime functions have expression kind == OTHER_FUNCTION, even if they
        // return an array
        if (!exp.getType().equals(expectedFieldType)
                && !(is_array_kind || is_func_kind)) {
          exp = builder.makeCast(expectedFieldType, exp);
        }
      }

      nodes.add(exp);
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

  RelRoot queryToRelNode(final String sql, final HeavyDBParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    final HeavyDBPlanner planner = getPlanner(
            true, parserOptions.isWatchdogEnabled(), parserOptions.isDistributedMode());
    final SqlNode sqlNode = parseSql(sql, parserOptions.isLegacySyntax(), planner);
    return convertSqlToRelNode(sqlNode, planner, parserOptions);
  }

  RelRoot convertSqlToRelNode(final SqlNode sqlNode,
          final HeavyDBPlanner HeavyDBPlanner,
          final HeavyDBParserOptions parserOptions)
          throws SqlParseException, ValidationException, RelConversionException {
    SqlNode node = sqlNode;
    HeavyDBPlanner planner = HeavyDBPlanner;
    boolean allowCorrelatedSubQueryExpansion = true;
    boolean patchUpdateToDelete = false;
    if (node.isA(DELETE)) {
      SqlDelete sqlDelete = (SqlDelete) node;
      node = new SqlUpdate(node.getParserPosition(),
              sqlDelete.getTargetTable(),
              SqlNodeList.EMPTY,
              SqlNodeList.EMPTY,
              sqlDelete.getCondition(),
              sqlDelete.getSourceSelect(),
              sqlDelete.getAlias());

      patchUpdateToDelete = true;
    }
    if (node.isA(UPDATE)) {
      SqlUpdate update = (SqlUpdate) node;
      update = (SqlUpdate) planner.validate(update);
      RelRoot root = rewriteUpdateAsSelect(update, parserOptions);

      if (patchUpdateToDelete) {
        LogicalTableModify modify = (LogicalTableModify) root.rel;

        try {
          Field f = TableModify.class.getDeclaredField("operation");
          f.setAccessible(true);
          f.set(modify, Operation.DELETE);
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }

        root = RelRoot.of(modify, SqlKind.DELETE);
      }

      return root;
    }
    if (parserOptions.isLegacySyntax()) {
      // close original planner
      planner.close();
      // create a new one
      planner = getPlanner(allowCorrelatedSubQueryExpansion,
              parserOptions.isWatchdogEnabled(),
              parserOptions.isDistributedMode());
      node = parseSql(
              node.toSqlString(CalciteSqlDialect.DEFAULT).toString(), false, planner);
    }

    SqlNode validateR = planner.validate(node);
    planner.setFilterPushDownInfo(parserOptions.getFilterPushDownInfo());
    // check to see if a view is involved in the query
    boolean foundView = false;
    SqlIdentifierCapturer capturer = captureIdentifiers(sqlNode);
    for (ImmutableList<String> names : capturer.selects) {
      HeavyDBSchema schema = new HeavyDBSchema(
              dataDir, this, dbPort, dbUser, sock_transport_properties, names.get(1));
      HeavyDBTable table = (HeavyDBTable) schema.getTable(names.get(0));
      if (null == table) {
        throw new RuntimeException("table/view not found: " + names.get(0));
      }
      if (table instanceof HeavyDBView) {
        foundView = true;
      }
    }
    RelRoot relRootNode = planner.getRelRoot(validateR);
    relRootNode = replaceIsTrue(planner.getTypeFactory(), relRootNode);
    RelNode rootNode = planner.optimizeRATree(
            relRootNode.project(), parserOptions.isViewOptimizeEnabled(), foundView);
    planner.close();
    return new RelRoot(rootNode,
            relRootNode.validatedRowType,
            relRootNode.kind,
            relRootNode.fields,
            relRootNode.collation,
            Collections.emptyList());
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

    return new RelRoot(node,
            root.validatedRowType,
            root.kind,
            root.fields,
            root.collation,
            Collections.emptyList());
  }

  private SqlNode parseSql(String sql, final boolean legacy_syntax, Planner planner)
          throws SqlParseException {
    SqlNode parseR = null;
    try {
      parseR = planner.parse(sql);
      HEAVYDBLOGGER.debug(" node is \n" + parseR.toString());
    } catch (SqlParseException ex) {
      HEAVYDBLOGGER.error("failed to parse SQL '" + sql + "' \n" + ex.toString());
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
      } else if (order_by_node.query instanceof SqlWith) {
        SqlWith old_with_node = (SqlWith) order_by_node.query;
        if (old_with_node.body instanceof SqlSelect) {
          select_node = (SqlSelect) old_with_node.body;
          desugar(select_node, typeFactory);
        }
      }
    } else if (parseR instanceof SqlWith) {
      SqlWith old_with_node = (SqlWith) parseR;
      if (old_with_node.body instanceof SqlSelect) {
        select_node = (SqlSelect) old_with_node.body;
        desugar(select_node, typeFactory);
      }
    }
    return parseR;
  }

  private void desugar(SqlSelect select_node, RelDataTypeFactory typeFactory) {
    desugar(select_node, null, typeFactory);
  }

  private SqlNode expandCase(SqlCase old_case_node, RelDataTypeFactory typeFactory) {
    SqlNodeList newWhenList =
            new SqlNodeList(old_case_node.getWhenOperands().getParserPosition());
    SqlNodeList newThenList =
            new SqlNodeList(old_case_node.getThenOperands().getParserPosition());
    java.util.Map<String, SqlNode> id_to_expr = new java.util.HashMap<String, SqlNode>();
    for (SqlNode node : old_case_node.getWhenOperands()) {
      SqlNode newCall = expand(node, id_to_expr, typeFactory);
      if (null != newCall) {
        newWhenList.add(newCall);
      } else {
        newWhenList.add(node);
      }
    }
    for (SqlNode node : old_case_node.getThenOperands()) {
      SqlNode newCall = expand(node, id_to_expr, typeFactory);
      if (null != newCall) {
        newThenList.add(newCall);
      } else {
        newThenList.add(node);
      }
    }
    SqlNode new_else_operand = old_case_node.getElseOperand();
    if (null != new_else_operand) {
      SqlNode candidate_else_operand =
              expand(old_case_node.getElseOperand(), id_to_expr, typeFactory);
      if (null != candidate_else_operand) {
        new_else_operand = candidate_else_operand;
      }
    }
    SqlNode new_value_operand = old_case_node.getValueOperand();
    if (null != new_value_operand) {
      SqlNode candidate_value_operand =
              expand(old_case_node.getValueOperand(), id_to_expr, typeFactory);
      if (null != candidate_value_operand) {
        new_value_operand = candidate_value_operand;
      }
    }
    SqlNode newCaseNode = SqlCase.createSwitched(old_case_node.getParserPosition(),
            new_value_operand,
            newWhenList,
            newThenList,
            new_else_operand);
    return newCaseNode;
  }

  private SqlOrderBy desugar(SqlSelect select_node,
          SqlOrderBy order_by_node,
          RelDataTypeFactory typeFactory) {
    HEAVYDBLOGGER.debug("desugar: before: " + select_node.toString());
    desugarExpression(select_node.getFrom(), typeFactory);
    desugarExpression(select_node.getWhere(), typeFactory);
    SqlNodeList select_list = select_node.getSelectList();
    SqlNodeList new_select_list = new SqlNodeList(select_list.getParserPosition());
    java.util.Map<String, SqlNode> id_to_expr = new java.util.HashMap<String, SqlNode>();
    for (SqlNode proj : select_list) {
      if (!(proj instanceof SqlBasicCall)) {
        if (proj instanceof SqlCase) {
          new_select_list.add(expandCase((SqlCase) proj, typeFactory));
        } else {
          new_select_list.add(proj);
        }
      } else {
        assert proj instanceof SqlBasicCall;
        SqlBasicCall proj_call = (SqlBasicCall) proj;
        if (proj_call.operands.length > 0) {
          for (int i = 0; i < proj_call.operands.length; i++) {
            if (proj_call.operand(i) instanceof SqlCase) {
              SqlNode new_op = expandCase(proj_call.operand(i), typeFactory);
              proj_call.setOperand(i, new_op);
            }
          }
        }
        new_select_list.add(expand(proj_call, id_to_expr, typeFactory));
      }
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

    HEAVYDBLOGGER.debug("desugar:  after: " + select_node.toString());
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
    HEAVYDBLOGGER.debug("expand: " + node.toString());
    if (node instanceof SqlBasicCall) {
      SqlBasicCall node_call = (SqlBasicCall) node;
      SqlNode[] operands = node_call.getOperands();
      for (int i = 0; i < operands.length; ++i) {
        node_call.setOperand(i, expand(operands[i], id_to_expr, typeFactory));
      }
      SqlNode expanded_string_function = expandStringFunctions(node_call, typeFactory);
      if (expanded_string_function != null) {
        return expanded_string_function;
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

  private SqlNode expandStringFunctions(
          final SqlBasicCall proj_call, RelDataTypeFactory typeFactory) {
    //
    // Expand string functions
    //

    final int operandCount = proj_call.operandCount();

    if (proj_call.getOperator().isName("MID", false)
            || proj_call.getOperator().isName("SUBSTR", false)) {
      // Replace MID/SUBSTR with SUBSTRING
      //
      // Note: SUBSTRING doesn't offer much flexibility for the numeric arg's type
      //    "only constant, column, or other string operator arguments are allowed"
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 2) {
        final SqlNode primary_operand = proj_call.operand(0);
        final SqlNode from_operand = proj_call.operand(1);
        return SqlStdOperatorTable.SUBSTRING.createCall(
                pos, primary_operand, from_operand);

      } else if (operandCount == 3) {
        final SqlNode primary_operand = proj_call.operand(0);
        final SqlNode from_operand = proj_call.operand(1);
        final SqlNode for_operand = proj_call.operand(2);
        return SqlStdOperatorTable.SUBSTRING.createCall(
                pos, primary_operand, from_operand, for_operand);
      }
      return null;

    } else if (proj_call.getOperator().isName("CONTAINS", false)) {
      // Replace CONTAINS with LIKE
      //     as noted by TABLEAU's own published documention
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 2) {
        final SqlNode primary = proj_call.operand(0);
        final SqlNode pattern = proj_call.operand(1);

        if (pattern instanceof SqlLiteral) {
          // LIKE only supports Literal patterns ... at the moment
          SqlLiteral literalPattern = (SqlLiteral) pattern;
          String sPattern = literalPattern.getValueAs(String.class);
          SqlLiteral withWildcards =
                  SqlLiteral.createCharString("%" + sPattern + "%", pos);
          return SqlStdOperatorTable.LIKE.createCall(pos, primary, withWildcards);
        }
      }
      return null;

    } else if (proj_call.getOperator().isName("ENDSWITH", false)) {
      // Replace ENDSWITH with LIKE
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 2) {
        final SqlNode primary = proj_call.operand(0);
        final SqlNode pattern = proj_call.operand(1);

        if (pattern instanceof SqlLiteral) {
          // LIKE only supports Literal patterns ... at the moment
          SqlLiteral literalPattern = (SqlLiteral) pattern;
          String sPattern = literalPattern.getValueAs(String.class);
          SqlLiteral withWildcards = SqlLiteral.createCharString("%" + sPattern, pos);
          return SqlStdOperatorTable.LIKE.createCall(pos, primary, withWildcards);
        }
      }
      return null;
    } else if (proj_call.getOperator().isName("LCASE", false)) {
      // Expand LCASE with LOWER
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 1) {
        final SqlNode primary = proj_call.operand(0);
        return SqlStdOperatorTable.LOWER.createCall(pos, primary);
      }
      return null;

    } else if (proj_call.getOperator().isName("LEFT", false)) {
      // Replace LEFT with SUBSTRING
      final SqlParserPos pos = proj_call.getParserPosition();

      if (operandCount == 2) {
        final SqlNode primary = proj_call.operand(0);
        SqlNode start = SqlLiteral.createExactNumeric("0", SqlParserPos.ZERO);
        final SqlNode count = proj_call.operand(1);
        return SqlStdOperatorTable.SUBSTRING.createCall(pos, primary, start, count);
      }
      return null;

    } else if (proj_call.getOperator().isName("LEN", false)) {
      // Replace LEN with CHARACTER_LENGTH
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 1) {
        final SqlNode primary = proj_call.operand(0);
        return SqlStdOperatorTable.CHARACTER_LENGTH.createCall(pos, primary);
      }
      return null;

    } else if (proj_call.getOperator().isName("MAX", false)
            || proj_call.getOperator().isName("MIN", false)) {
      // Replace MAX(a,b), MIN(a,b) with CASE
      final SqlParserPos pos = proj_call.getParserPosition();

      if (operandCount == 2) {
        final SqlNode arg1 = proj_call.operand(0);
        final SqlNode arg2 = proj_call.operand(1);

        SqlNodeList whenList = new SqlNodeList(pos);
        SqlNodeList thenList = new SqlNodeList(pos);
        SqlNodeList elseClause = new SqlNodeList(pos);

        if (proj_call.getOperator().isName("MAX", false)) {
          whenList.add(
                  SqlStdOperatorTable.GREATER_THAN_OR_EQUAL.createCall(pos, arg1, arg2));
        } else {
          whenList.add(
                  SqlStdOperatorTable.LESS_THAN_OR_EQUAL.createCall(pos, arg1, arg2));
        }
        thenList.add(arg1);
        elseClause.add(arg2);

        SqlNode caseIdentifier = null;
        return SqlCase.createSwitched(
                pos, caseIdentifier, whenList, thenList, elseClause);
      }
      return null;

    } else if (proj_call.getOperator().isName("RIGHT", false)) {
      // Replace RIGHT with SUBSTRING
      final SqlParserPos pos = proj_call.getParserPosition();

      if (operandCount == 2) {
        final SqlNode primary = proj_call.operand(0);
        final SqlNode count = proj_call.operand(1);
        if (count instanceof SqlNumericLiteral) {
          SqlNumericLiteral numericCount = (SqlNumericLiteral) count;
          if (numericCount.intValue(true) > 0) {
            // common case
            final SqlNode negativeCount =
                    SqlNumericLiteral.createNegative(numericCount, pos);
            return SqlStdOperatorTable.SUBSTRING.createCall(pos, primary, negativeCount);
          }
          // allow zero (or negative) to return an empty string
          //   matches behavior of LEFT
          SqlNode zero = SqlLiteral.createExactNumeric("0", SqlParserPos.ZERO);
          return SqlStdOperatorTable.SUBSTRING.createCall(pos, primary, zero, zero);
        }
        // if not a simple literal ... attempt to evaluate
        //    expected to fail ... with a useful error message
        return SqlStdOperatorTable.SUBSTRING.createCall(pos, primary, count);
      }
      return null;

    } else if (proj_call.getOperator().isName("SPACE", false)) {
      // Replace SPACE with REPEAT
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 1) {
        final SqlNode count = proj_call.operand(0);
        SqlFunction fn_repeat = new SqlFunction("REPEAT",
                SqlKind.OTHER_FUNCTION,
                ReturnTypes.ARG0_NULLABLE,
                null,
                OperandTypes.CHARACTER,
                SqlFunctionCategory.STRING);
        SqlLiteral space = SqlLiteral.createCharString(" ", pos);
        return fn_repeat.createCall(pos, space, count);
      }
      return null;

    } else if (proj_call.getOperator().isName("SPLIT", false)) {
      // Replace SPLIT with SPLIT_PART
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 3) {
        final SqlNode primary = proj_call.operand(0);
        final SqlNode delimeter = proj_call.operand(1);
        final SqlNode count = proj_call.operand(2);
        SqlFunction fn_split = new SqlFunction("SPLIT_PART",
                SqlKind.OTHER_FUNCTION,
                ReturnTypes.ARG0_NULLABLE,
                null,
                OperandTypes.CHARACTER,
                SqlFunctionCategory.STRING);

        return fn_split.createCall(pos, primary, delimeter, count);
      }
      return null;

    } else if (proj_call.getOperator().isName("STARTSWITH", false)) {
      // Replace STARTSWITH with LIKE
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 2) {
        final SqlNode primary = proj_call.operand(0);
        final SqlNode pattern = proj_call.operand(1);

        if (pattern instanceof SqlLiteral) {
          // LIKE only supports Literal patterns ... at the moment
          SqlLiteral literalPattern = (SqlLiteral) pattern;
          String sPattern = literalPattern.getValueAs(String.class);
          SqlLiteral withWildcards = SqlLiteral.createCharString(sPattern + "%", pos);
          return SqlStdOperatorTable.LIKE.createCall(pos, primary, withWildcards);
        }
      }
      return null;

    } else if (proj_call.getOperator().isName("UCASE", false)) {
      // Replace UCASE with UPPER
      final SqlParserPos pos = proj_call.getParserPosition();
      if (operandCount == 1) {
        final SqlNode primary = proj_call.operand(0);
        return SqlStdOperatorTable.UPPER.createCall(pos, primary);
      }
      return null;
    }

    return null;
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
    HEAVYDBLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
    HEAVYDBLOGGER.debug("to : " + expanded_proj_call.toString());
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
    HEAVYDBLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
    HEAVYDBLOGGER.debug("to : " + expanded_proj_call.toString());
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
    HEAVYDBLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
    HEAVYDBLOGGER.debug("to : " + expanded_proj_call.toString());
    return expanded_proj_call;
  }

  public SqlIdentifierCapturer captureIdentifiers(String sql, boolean legacy_syntax)
          throws SqlParseException {
    try {
      Planner planner = getPlanner();
      SqlNode node = parseSql(sql, legacy_syntax, planner);
      return captureIdentifiers(node);
    } catch (Exception | Error e) {
      HEAVYDBLOGGER.error("Error parsing sql: " + sql, e);
      return new SqlIdentifierCapturer();
    }
  }

  public SqlIdentifierCapturer captureIdentifiers(SqlNode node) throws SqlParseException {
    try {
      SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
      capturer.scan(node);
      capturer.selects = addDbContextIfMissing(capturer.selects);
      capturer.updates = addDbContextIfMissing(capturer.updates);
      capturer.deletes = addDbContextIfMissing(capturer.deletes);
      capturer.inserts = addDbContextIfMissing(capturer.inserts);
      return capturer;
    } catch (Exception | Error e) {
      HEAVYDBLOGGER.error("Error parsing sql: " + node, e);
      return new SqlIdentifierCapturer();
    }
  }

  private Set<ImmutableList<String>> addDbContextIfMissing(
          Set<ImmutableList<String>> names) {
    Set<ImmutableList<String>> result = new HashSet<>();
    for (ImmutableList<String> name : names) {
      if (name.size() == 1) {
        result.add(new ImmutableList.Builder<String>()
                           .addAll(name)
                           .add(dbUser.getDB())
                           .build());
      } else {
        result.add(name);
      }
    }
    return result;
  }

  public int getCallCount() {
    return callCount;
  }

  public void updateMetaData(String schema, String table) {
    HEAVYDBLOGGER.debug("schema :" + schema + " table :" + table);
    HeavyDBSchema db = new HeavyDBSchema(
            dataDir, this, dbPort, null, sock_transport_properties, schema);
    db.updateMetaData(schema, table);
  }

  protected RelDataTypeSystem createTypeSystem() {
    final HeavyDBTypeSystem typeSystem = new HeavyDBTypeSystem();
    return typeSystem;
  }

  private static class ExpressionListedInSelectClauseChecker
          extends SqlBasicVisitor<Void> {
    @Override
    public Void visit(SqlCall call) {
      if (call instanceof SqlSelect) {
        SqlSelect selectNode = (SqlSelect) call;
        String targetString = targetExpression.toString();
        for (SqlNode listedNode : selectNode.getSelectList()) {
          if (listedNode.toString().contains(targetString)) {
            throw Util.FoundOne.NULL;
          }
        }
      }
      return super.visit(call);
    }

    boolean containsExpression(SqlNode node, SqlNode targetExpression) {
      try {
        this.targetExpression = targetExpression;
        node.accept(this);
        return false;
      } catch (Util.FoundOne e) {
        return true;
      }
    }

    SqlNode targetExpression;
  }

  private static class ExpressionListedAsChildOROperatorChecker
          extends SqlBasicVisitor<Void> {
    @Override
    public Void visit(SqlCall call) {
      if (call instanceof SqlBasicCall) {
        SqlBasicCall basicCall = (SqlBasicCall) call;
        if (basicCall.getKind() == SqlKind.OR) {
          String targetString = targetExpression.toString();
          for (SqlNode listedOperand : basicCall.operands) {
            if (listedOperand.toString().contains(targetString)) {
              throw Util.FoundOne.NULL;
            }
          }
        }
      }
      return super.visit(call);
    }

    boolean containsExpression(SqlNode node, SqlNode targetExpression) {
      try {
        this.targetExpression = targetExpression;
        node.accept(this);
        return false;
      } catch (Util.FoundOne e) {
        return true;
      }
    }

    SqlNode targetExpression;
  }

  private static class JoinOperatorChecker extends SqlBasicVisitor<Void> {
    Set<SqlBasicCall> targetCalls = new HashSet<>();

    public boolean isEqualityJoinOperator(SqlBasicCall basicCall) {
      if (null != basicCall) {
        if (basicCall.operands.length == 2
                && (basicCall.getKind() == SqlKind.EQUALS
                        || basicCall.getKind() == SqlKind.NOT_EQUALS)
                && basicCall.operand(0) instanceof SqlIdentifier
                && basicCall.operand(1) instanceof SqlIdentifier) {
          return true;
        }
      }
      return false;
    }

    @Override
    public Void visit(SqlCall call) {
      if (call instanceof SqlBasicCall) {
        targetCalls.add((SqlBasicCall) call);
      }
      for (SqlNode node : call.getOperandList()) {
        if (null != node && !targetCalls.contains(node)) {
          node.accept(this);
        }
      }
      return super.visit(call);
    }

    boolean containsExpression(SqlNode node) {
      try {
        if (null != node) {
          node.accept(this);
          for (SqlBasicCall basicCall : targetCalls) {
            if (isEqualityJoinOperator(basicCall)) {
              throw Util.FoundOne.NULL;
            }
          }
        }
        return false;
      } catch (Util.FoundOne e) {
        return true;
      }
    }
  }

  // this visitor checks whether a parse tree contains at least one
  // specific SQL operator we have an interest in
  // (do not count the accurate # operators we found)
  private static class FindSqlOperator extends SqlBasicVisitor<Void> {
    @Override
    public Void visit(SqlCall call) {
      if (call instanceof SqlBasicCall) {
        SqlBasicCall basicCall = (SqlBasicCall) call;
        if (basicCall.getKind().equals(targetKind)) {
          throw Util.FoundOne.NULL;
        }
      }
      return super.visit(call);
    }

    boolean containsSqlOperator(SqlNode node, SqlKind operatorKind) {
      try {
        targetKind = operatorKind;
        node.accept(this);
        return false;
      } catch (Util.FoundOne e) {
        return true;
      }
    }

    private SqlKind targetKind;
  }

  public void tableAliasFinder(SqlNode sqlNode, Map<String, String> tableAliasMap) {
    final SqlVisitor<Void> aliasCollector = new SqlBasicVisitor<Void>() {
      @Override
      public Void visit(SqlCall call) {
        if (call instanceof SqlBasicCall) {
          SqlBasicCall basicCall = (SqlBasicCall) call;
          if (basicCall.getKind() == SqlKind.AS) {
            if (basicCall.operand(0) instanceof SqlIdentifier) {
              // we need to check whether basicCall's the first operand is SqlIdentifier
              // since sometimes it represents non column identifier like SqlSelect
              SqlIdentifier colNameIdentifier = (SqlIdentifier) basicCall.operand(0);
              String tblName = colNameIdentifier.names.size() == 1
                      ? colNameIdentifier.names.get(0)
                      : colNameIdentifier.names.get(1);
              tableAliasMap.put(basicCall.operand(1).toString(), tblName);
            }
          }
        }
        return super.visit(call);
      }
    };
    sqlNode.accept(aliasCollector);
  }
}

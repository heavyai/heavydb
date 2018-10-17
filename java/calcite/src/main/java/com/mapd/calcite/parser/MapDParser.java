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
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.MapDPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlAsOperator;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.apache.calcite.util.ConversionUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mapd.parser.server.ExtensionFunction;
import com.mapd.common.SockTransportProperties;
/**
 *
 * @author michael
 */
public final class MapDParser {
  public static final ThreadLocal<MapDParser> CURRENT_PARSER = new ThreadLocal<>();

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDParser.class);

  //    private SqlTypeFactoryImpl typeFactory;
  //    private MapDCatalogReader catalogReader;
  //    private SqlValidatorImpl validator;
  //    private SqlToRelConverter converter;
  private final Map<String, ExtensionFunction> extSigs;
  private final String dataDir;

  private int callCount = 0;
  private final int mapdPort;
  private MapDUser mapdUser;
  private SockTransportProperties sock_transport_properties = null;
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

  private MapDPlanner getPlanner() {
    MapDSchema mapd =
            new MapDSchema(dataDir, this, mapdPort, mapdUser, sock_transport_properties);
    final SchemaPlus rootSchema = Frameworks.createRootSchema(true);
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(rootSchema.add(mapdUser.getDB(), mapd))
                    .operatorTable(createOperatorTable(extSigs))
                    .parserConfig(SqlParser.configBuilder()
                                          .setUnquotedCasing(Casing.UNCHANGED)
                                          .setCaseSensitive(false)
                                          .build())
                    .build();
    return new MapDPlanner(config);
  }

  public void setUser(MapDUser mapdUser) {
    this.mapdUser = mapdUser;
  }

  public static class FilterPushDownInfo {
    public FilterPushDownInfo(
            final int input_prev, final int input_start, final int input_next) {
      this.input_prev = input_prev;
      this.input_start = input_start;
      this.input_next = input_next;
    }

    public int input_prev;
    public int input_start;
    public int input_next;
  }

  public String getRelAlgebra(String sql,
          final List<FilterPushDownInfo> filterPushDownInfo,
          final boolean legacy_syntax,
          final MapDUser mapDUser,
          final boolean isExplain)
          throws SqlParseException, ValidationException, RelConversionException {
    callCount++;
    final RelRoot sqlRel = queryToSqlNode(sql, filterPushDownInfo, legacy_syntax);

    RelNode project = sqlRel.project();

    if (isExplain) {
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

  RelRoot queryToSqlNode(final String sql,
          final List<FilterPushDownInfo> filterPushDownInfo,
          final boolean legacy_syntax)
          throws SqlParseException, ValidationException, RelConversionException {
    MapDPlanner planner = getPlanner();

    SqlNode node = processSQL(sql, legacy_syntax, planner);
    if (legacy_syntax) {
      // close original planner
      planner.close();
      // create a new one
      planner = getPlanner();
      node = processSQL(node.toSqlString(SqlDialect.CALCITE).toString(), false, planner);
    }

    boolean is_select_star = isSelectStar(node);

    SqlNode validateR = planner.validate(node);
    SqlSelect validate_select = getSelectChild(validateR);

    // Hide rowid from select * queries
    if (legacy_syntax && is_select_star && validate_select != null) {
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
      planner = getPlanner();
      processSQL(validateR.toSqlString(SqlDialect.CALCITE).toString(), false, planner);
      // now validate the new modified SqlNode;
      validateR = planner.validate(validateR);
    }

    planner.setFilterPushDownInfo(filterPushDownInfo);
    RelRoot relR = planner.rel(validateR);
    planner.close();
    return relR;
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
    if (proj_call.getOperator().isName("STDDEV_POP")) {
      biased = true;
      sqrt = true;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("STDDEV_POP_FLOAT")) {
      biased = true;
      sqrt = true;
      flt = true;
    } else if (proj_call.getOperator().isName("STDDEV_SAMP")
            || proj_call.getOperator().getName().equalsIgnoreCase("STDDEV")) {
      biased = false;
      sqrt = true;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("STDDEV_SAMP_FLOAT")
            || proj_call.getOperator().getName().equalsIgnoreCase("STDDEV_FLOAT")) {
      biased = false;
      sqrt = true;
      flt = true;
    } else if (proj_call.getOperator().isName("VAR_POP")) {
      biased = true;
      sqrt = false;
      flt = false;
    } else if (proj_call.getOperator().getName().equalsIgnoreCase("VAR_POP_FLOAT")) {
      biased = true;
      sqrt = false;
      flt = true;
    } else if (proj_call.getOperator().isName("VAR_SAMP")
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
    //   power(
    //     (sum(x * x) - sum(x) * sum(x) / (case count(x) when 0 then NULL else count(x)
    //     end)) / (case count(x) when 0 then NULL else count(x) end), .5)
    //
    // stddev_samp(x) ==>
    //   power(
    //     (sum(x * x) - sum(x) * sum(x) / (case count(x) when 0 then NULL else count(x)
    //     )) / ((case count(x) when 1 then NULL else count(x) - 1 end)), .5)
    //
    // var_pop(x) ==>
    //     (sum(x * x) - sum(x) * sum(x) / ((case count(x) when 0 then NULL else count(x)
    //     end))) / ((case count(x) when 0 then NULL else count(x) end))
    //
    // var_samp(x) ==>
    //     (sum(x * x) - sum(x) * sum(x) / ((case count(x) when 0 then NULL else count(x)
    //     end))) / ((case count(x) when 1 then NULL else count(x) - 1 end))
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
    if (proj_call.getOperator().isName("COVAR_POP")) {
      pop = true;
      flt = false;
    } else if (proj_call.getOperator().isName("COVAR_SAMP")) {
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
    //                      ((case count(x) when 1 then NULL else count(x) - 1 end))
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
    if (proj_call.getOperator().isName("CORR")
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
    // corr(x, y) ==> (avg(x * y) - avg(x) * avg(y)) / (stddev_pop(x) * stddev_pop(y))
    //            ==> covar_pop(x, y) / (stddev_pop(x) * stddev_pop(y))
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

  public int getCallCount() {
    return callCount;
  }

  public void updateMetaData(String schema, String table) {
    MAPDLOGGER.debug("schema :" + schema + " table :" + table);
    MapDSchema mapd =
            new MapDSchema(dataDir, this, mapdPort, null, sock_transport_properties);
    mapd.updateMetaData(schema, table);
  }
}

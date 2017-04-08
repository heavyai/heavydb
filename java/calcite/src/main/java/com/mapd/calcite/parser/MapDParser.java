package com.mapd.calcite.parser;

import com.mapd.parser.server.ExtensionFunction;
import java.util.List;
import java.util.Map;
import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.sql.SqlAsOperator;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeFactoryImpl;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.sql2rel.SqlToRelConverter.Config;
import org.apache.calcite.sql2rel.StandardConvertletTable;
import org.apache.calcite.util.ConversionUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public final class MapDParser {

    final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDParser.class);

    private SqlTypeFactoryImpl typeFactory;
    private MapDCatalogReader catalogReader;
    private SqlValidatorImpl validator;
    private SqlToRelConverter converter;
    private final Map<String, ExtensionFunction> extSigs;
    private final String dataDir;

    private int callCount = 0;

    public MapDParser(String dataDir, final Map<String, ExtensionFunction> extSigs) {
        System.setProperty("saffron.default.charset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
        System.setProperty("saffron.default.nationalcharset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
        System.setProperty("saffron.default.collation.name", ConversionUtil.NATIVE_UTF16_CHARSET_NAME + "$en_US");
        this.dataDir = dataDir;
        this.extSigs = extSigs;
    }

    public class Expander implements RelOptTable.ViewExpander {

        @Override
        public RelRoot expandView(RelDataType rowType, String queryString,
                List<String> schemaPath, List<String> viewPath) {
            try {
                return queryToSqlNode(queryString, true);
            } catch (SqlParseException e) {
                return null;
            }
        }
    }

    public String getRelAlgebra(String sql, final boolean legacy_syntax, final MapDUser mapDUser, final boolean isExplain)
            throws SqlParseException {
        callCount++;
        catalogReader = new MapDCatalogReader(new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT), dataDir, this);
        catalogReader.setCurrentMapDUser(mapDUser);
        final RelRoot sqlRel = queryToSqlNode(sql, legacy_syntax);
        RelNode project = sqlRel.project();

        if (isExplain) {
            return RelOptUtil.toString(sqlRel.project());
        }

        String res = MapDSerializer.toString(project);

        return res;
    }

    RelRoot queryToSqlNode(final String sql, final boolean legacy_syntax) throws SqlParseException {
        typeFactory = new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT);

        SqlNode node = processSQL(sql, legacy_syntax);
        if (legacy_syntax) {
            node = processSQL(node.toSqlString(SqlDialect.CALCITE).toString(), false);
        }

        boolean is_select_star = isSelectStar(node);

        validator = new MapDValidator(
                createOperatorTable(extSigs),
                catalogReader,
                typeFactory,
                SqlConformance.DEFAULT);

        SqlNode validate = validator.validate(node);

        SqlSelect validate_select = getSelectChild(validate);

        // Hide rowid from select * queries
        if (legacy_syntax && is_select_star && validate_select != null) {
            SqlNodeList proj_exprs = ((SqlSelect) validate).getSelectList();
            SqlNodeList new_proj_exprs = new SqlNodeList(proj_exprs.getParserPosition());
            for (SqlNode proj_expr : proj_exprs) {
                final SqlNode unaliased_proj_expr = getUnaliasedExpression(proj_expr);
                if (unaliased_proj_expr instanceof SqlIdentifier
                        && (((SqlIdentifier) unaliased_proj_expr).toString().toLowerCase()).endsWith(".rowid")) {
                    continue;
                }
                new_proj_exprs.add(proj_expr);
            }
            validate_select.setSelectList(new_proj_exprs);
        }

        final RexBuilder rexBuilder = new RexBuilder(typeFactory);
        final RelOptCluster cluster = RelOptCluster.create(new MapDRelOptPlanner(), rexBuilder);
        final Config config = SqlToRelConverter.configBuilder().withExpand(false).withInSubQueryThreshold(Integer.MAX_VALUE).build();
        converter = new SqlToRelConverter(new Expander(), validator, catalogReader, cluster,
                StandardConvertletTable.INSTANCE, config);

        return converter.convertQuery(validate, true, true);
    }

    private static SqlNode getUnaliasedExpression(final SqlNode node) {
        if (node instanceof SqlBasicCall && ((SqlBasicCall) node).getOperator() instanceof SqlAsOperator) {
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

    private SqlNode processSQL(String sql, final boolean legacy_syntax) throws SqlParseException {
        SqlNode node = null;
        SqlParser sqlp = getSqlParser(sql);
        try {
            node = sqlp.parseStmt();
            MAPDLOGGER.debug(" node is \n" + node.toString());
        } catch (SqlParseException ex) {
            MAPDLOGGER.error("failed to process SQL '" + sql + "' \n" + ex.toString());
            throw ex;
        }
        if (!legacy_syntax) {
            return node;
        }
        SqlSelect select_node = null;
        if (node instanceof SqlSelect) {
            select_node = (SqlSelect) node;
        } else if (node instanceof SqlOrderBy) {
            SqlOrderBy order_by_node = (SqlOrderBy) node;
            if (order_by_node.query instanceof SqlSelect) {
                select_node = (SqlSelect) order_by_node.query;
            }
        }
        if (select_node != null) {
            desugar(select_node);
        }
        return node;
    }

    private void desugar(SqlSelect select_node) {
        MAPDLOGGER.debug("desugar: before: " + select_node.toString());
        desugarExpression(select_node.getFrom());
        desugarExpression(select_node.getWhere());
        SqlNodeList select_list = select_node.getSelectList();
        SqlNodeList new_select_list = new SqlNodeList(select_list.getParserPosition());
        java.util.Map<String, SqlNode> id_to_expr = new java.util.HashMap<String, SqlNode>();
        for (SqlNode proj : select_list) {
            if (!(proj instanceof SqlBasicCall)) {
                new_select_list.add(proj);
                continue;
            }
            SqlBasicCall proj_call = (SqlBasicCall) proj;
            if (proj_call.getOperator() instanceof SqlAsOperator) {
                MAPDLOGGER.debug("desugar: SqlBasicCall: " + proj_call.toString());
                SqlNode[] operands = proj_call.getOperands();
                SqlIdentifier id = (SqlIdentifier) operands[1];
                id_to_expr.put(id.toString(), operands[0]);
                new_select_list.add(proj_call);
                continue;
            }
            new_select_list.add(expand(proj_call, id_to_expr));
        }
        select_node.setSelectList(new_select_list);
        SqlNodeList group_by_list = select_node.getGroup();
        if (group_by_list != null) {
            select_node.setGroupBy(expand(group_by_list, id_to_expr));
        }
        SqlNode having = select_node.getHaving();
        if (having != null) {
            expand(having, id_to_expr);
        }

        MAPDLOGGER.debug("desugar:  after: " + select_node.toString());
    }

    private void desugarExpression(SqlNode node) {
        if (node instanceof SqlSelect) {
            desugar((SqlSelect) node);
            return;
        }
        if (!(node instanceof SqlBasicCall)) {
            return;
        }
        SqlBasicCall basic_call = (SqlBasicCall) node;
        for (SqlNode operator : basic_call.getOperands()) {
            if (operator instanceof SqlOrderBy) {
                desugarExpression(((SqlOrderBy) operator).query);
            } else {
                desugarExpression(operator);
            }
        }
    }

    private SqlNode expand(final SqlNode node,
            final java.util.Map<String, SqlNode> id_to_expr) {
        MAPDLOGGER.debug("expand: " + node.toString());
        if (node instanceof SqlIdentifier && id_to_expr.containsKey(node.toString())) {
            // Expand aliases
            return id_to_expr.get(node.toString());
        }
        if (node instanceof SqlBasicCall) {
            SqlBasicCall node_call = (SqlBasicCall) node;
            SqlNode[] operands = node_call.getOperands();
            for (int i = 0; i < operands.length; ++i) {
                node_call.setOperand(i, expand(operands[i], id_to_expr));
            }
            SqlNode expanded_variance = expandVariance(node_call);
            if (expanded_variance != null) {
                return expanded_variance;
            }
        }
        return node;
    }

    private SqlNodeList expand(final SqlNodeList group_by_list, final java.util.Map<String, SqlNode> id_to_expr) {
        SqlNodeList new_group_by_list = new SqlNodeList(new SqlParserPos(-1, -1));
        for (SqlNode group_by : group_by_list) {
            if (!(group_by instanceof SqlIdentifier)) {
                new_group_by_list.add(expand(group_by, id_to_expr));
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

    private SqlNode expandVariance(final SqlBasicCall proj_call) {
        // Expand variance aggregates that are not supported natively
        if (proj_call.operandCount() != 1) {
            return null;
        }
        boolean biased;
        boolean sqrt;
        if (proj_call.getOperator().isName("STDDEV_POP")) {
            biased = true;
            sqrt = true;
        } else if (proj_call.getOperator().isName("STDDEV")
                || proj_call.getOperator().isName("STDDEV_SAMP")) {
            biased = false;
            sqrt = true;
        } else if (proj_call.getOperator().isName("VAR_POP")) {
            biased = true;
            sqrt = false;
        } else if (proj_call.getOperator().isName("VAR_SAMP")
                || proj_call.getOperator().isName("VARIANCE")) {
            biased = false;
            sqrt = false;
        } else {
            return null;
        }
        final SqlNode operand = proj_call.operand(0);
        SqlNode expanded_proj_call = expandVariance(operand, biased, sqrt);
        MAPDLOGGER.debug("Expanded select_list SqlCall: " + proj_call.toString());
        MAPDLOGGER.debug("to : " + expanded_proj_call.toString());
        return expanded_proj_call;
    }

    private SqlNode expandVariance(final SqlNode operand, boolean biased, boolean sqrt) {
        // stddev_pop(x) ==>
        //   power(
        //     (sum(x * x) - sum(x) * sum(x) / (case count(x) when 0 then NULL else count(x) end))
        //     / (case count(x) when 0 then NULL else count(x) end),
        //     .5)
        //
        // stddev_samp(x) ==>
        //   power(
        //     (sum(x * x) - sum(x) * sum(x) / (case count(x) when 0 then NULL else count(x) ))
        //     / ((case count(x) when 1 then NULL else count(x) - 1 end)),
        //     .5)
        //
        // var_pop(x) ==>
        //     (sum(x * x) - sum(x) * sum(x) / ((case count(x) when 0 then NULL else count(x) end)))
        //     / ((case count(x) when 0 then NULL else count(x) end))
        //
        // var_samp(x) ==>
        //     (sum(x * x) - sum(x) * sum(x) / ((case count(x) when 0 then NULL else count(x) end)))
        //     / ((case count(x) when 1 then NULL else count(x) - 1 end))
        //
        final SqlParserPos pos = operand.getParserPosition();
        final SqlNode arg
                = SqlStdOperatorTable.CAST.createCall(pos, operand,
                        SqlTypeUtil.convertTypeToSpec(
                                typeFactory.createSqlType(SqlTypeName.DOUBLE)));
        final SqlNode argSquared
                = SqlStdOperatorTable.MULTIPLY.createCall(pos, arg, arg);
        final SqlNode sumArgSquared
                = SqlStdOperatorTable.SUM.createCall(pos, argSquared);
        final SqlNode sum
                = SqlStdOperatorTable.SUM.createCall(pos, arg);
        final SqlNode sumSquared
                = SqlStdOperatorTable.MULTIPLY.createCall(pos, sum, sum);
        final SqlNode count
                = SqlStdOperatorTable.COUNT.createCall(pos, arg);
        final SqlLiteral nul
                = SqlLiteral.createNull(pos);
        final SqlNumericLiteral zero
                = SqlLiteral.createExactNumeric("0", pos);
        final SqlNode countEqZero
                = SqlStdOperatorTable.EQUALS.createCall(pos, count, zero);
        SqlNodeList whenList = new SqlNodeList(pos);
        SqlNodeList thenList = new SqlNodeList(pos);
        whenList.add(countEqZero);
        thenList.add(nul);
        final SqlNode denominator
                = SqlStdOperatorTable.CASE.createCall(null, pos, null, whenList, thenList, count);
        final SqlNode avgSumSquared
                = SqlStdOperatorTable.DIVIDE.createCall(pos, sumSquared, denominator);
        final SqlNode diff
                = SqlStdOperatorTable.MINUS.createCall(pos, sumArgSquared, avgSumSquared);
        final SqlNode denominator1;
        if (biased) {
            denominator1 = denominator;
        } else {
            final SqlNumericLiteral one
                    = SqlLiteral.createExactNumeric("1", pos);
            final SqlNode countEqOne
                    = SqlStdOperatorTable.EQUALS.createCall(pos, count, one);
            final SqlNode countMinusOne
                    = SqlStdOperatorTable.MINUS.createCall(pos, count, one);
            SqlNodeList whenList1 = new SqlNodeList(pos);
            SqlNodeList thenList1 = new SqlNodeList(pos);
            whenList1.add(countEqOne);
            thenList1.add(nul);
            denominator1
                    = SqlStdOperatorTable.CASE.createCall(null, pos, null, whenList1, thenList1,
                            countMinusOne);
        }
        final SqlNode div
                = SqlStdOperatorTable.DIVIDE.createCall(pos, diff, denominator1);
        SqlNode result = div;
        if (sqrt) {
            final SqlNumericLiteral half
                    = SqlLiteral.createExactNumeric("0.5", pos);
            result
                    = SqlStdOperatorTable.POWER.createCall(pos, div, half);
        }
        return result;
    }

    /**
     * Creates an operator table.
     *
     * @param extSigs
     * @return New operator table
     */
    protected SqlOperatorTable createOperatorTable(final Map<String, ExtensionFunction> extSigs) {
        final MapDSqlOperatorTable tempOpTab
                = new MapDSqlOperatorTable(SqlStdOperatorTable.instance());
        // MAT 11 Nov 2015
        // Example of how to add custom function
        MapDSqlOperatorTable.addUDF(tempOpTab, extSigs);
        return tempOpTab;
    }

    protected SqlParser getSqlParser(String sql) {
        return SqlParser.create(sql,
                SqlParser.configBuilder()
                .setUnquotedCasing(Casing.UNCHANGED)
                .setCaseSensitive(false)
                .build());
    }

    public int getCallCount() {
        return callCount;
    }

    public void updateMetaData(String catalog, String table) {
        MAPDLOGGER.debug("catalog :" + catalog + " table :" + table);
        catalogReader = new MapDCatalogReader(new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT), dataDir, this);
        catalogReader.updateMetaData(catalog, table);
    }
}

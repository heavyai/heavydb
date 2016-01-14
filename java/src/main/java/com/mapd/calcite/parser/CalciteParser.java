/*
 * Some cool MapD Header
 */
package com.mapd.calcite.parser;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.avatica.util.Quoting;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.externalize.MapDRelJsonWriter;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.sql.SqlAsOperator;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeFactoryImpl;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorCatalogReader;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.sql2rel.StandardConvertletTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class MapDSerializer {

  static String toString(final RelNode rel) {
    if (rel == null) {
      return null;
    }
    final MapDRelJsonWriter planWriter = new MapDRelJsonWriter();
    rel.explain(planWriter);
    return planWriter.asString();
  }
}

public class CalciteParser {

  final static Logger logger = LoggerFactory.getLogger(CalciteParser.class);

  Quoting quoting = Quoting.DOUBLE_QUOTE;
  Casing unquotedCasing = Casing.UNCHANGED;
  Casing quotedCasing = Casing.UNCHANGED;

  private RelDataTypeFactory typeFactory;
  private Prepare.CatalogReader catalogReader;
  private SqlValidator validator;
  private SqlToRelConverter converter;

  private SqlOperatorTable opTab;
  private RelOptPlanner planner;

  public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException, IOException, SqlParseException {
    logger.info("Hello, World -- CalciteParser here");

    CalciteParser x = new CalciteParser();
    x.doWork(args);
  }

  protected void doWork(String[] args) throws UnsupportedEncodingException, FileNotFoundException, IOException, SqlParseException {
    logger.debug("In doWork");

    logger.info(getRelAlgebra("SELECT origin_lon, origin_lat FROM flights group by origin_lon, origin_lat", false));

    //logger.info(getRelAlgebra("Select * from (SELECT a.deptime*1.4 as delay, a.foodrequest, b.plane_engine_type, "
    //       + "b.weatherdelay FROM flights b join food a on a.deptime=b.deptime"
    //       + " where (a.deptime * 1.413) =1234343)"));/
    //logger.info(getRelAlgebra("SELECT * FROM SALES.EMP"));
    long timer = System.currentTimeMillis();
//        for (int i = 0; i < 100; i++) {
//            getRelAlgebra("select empno from emp");
//        }

    logger.info("time for 100 parses is " + (System.currentTimeMillis() - timer) + " ms");
  }

  public String getRelAlgebra(String sql, final boolean legacy_syntax) throws SqlParseException {
    long timer = System.currentTimeMillis();
    SqlNode node = processSQL(sql, legacy_syntax);

    typeFactory = getTypeFactory();

    final Prepare.CatalogReader catalogReader
            = createCatalogReader(typeFactory);

    final SqlValidator validator
            = createValidator(
                    catalogReader, typeFactory);

    SqlNode validate = validator.validate(node);

    final SqlToRelConverter converter
            = createSqlToRelConverter(
                    validator,
                    catalogReader,
                    typeFactory);

    final RelRoot sqlRel = converter.convertQuery(node, true, true);
    //final RelNode sqlRel = converter.convertSelect((SqlSelect)node, true);
    //RexNode convertExpression = converter.convertExpression(node);

    //logger.debug("After convert relNode is "+ convertExpression.toString());
    //logger.debug("After convert relRoot kind is " + sqlRel.kind);

    //logger.debug("After convert relRoot project is " + sqlRel.project().toString());

    //logger.debug("After convert relalgebra is \n" + RelOptUtil.toString(sqlRel.project()));

    RelNode project = sqlRel.project();

    String res = MapDSerializer.toString(project);

    //logger.info("After convert relalgebra is \n" + res);

    return res;
  }

  private SqlNode processSQL(String sql, final boolean legacy_syntax) throws SqlParseException {
    SqlNode node = null;
    SqlParser sqlp = getSqlParser(sql);
    try {
      node = sqlp.parseStmt();
      logger.debug(" node is \n" + node.toString());
    } catch (SqlParseException ex) {
      logger.error("failed to process SQL '" + sql + "' \n" + ex.toString());
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

  private static void desugar(SqlSelect select_node) {
    SqlNodeList select_list = select_node.getSelectList();
    java.util.Map<String, SqlNode> id_to_expr = new java.util.HashMap<String, SqlNode>();
    for (SqlNode proj : select_list) {
      if (!(proj instanceof SqlBasicCall)) {
        continue;
      }
      SqlBasicCall proj_call = (SqlBasicCall) proj;
      if (proj_call.getOperator() instanceof SqlAsOperator) {
        SqlNode[] operands = proj_call.getOperands();
        SqlIdentifier id = (SqlIdentifier) operands[1];
        id_to_expr.put(id.toString(), operands[0]);
      }
    }
    SqlNodeList group_by_list = select_node.getGroup();
    if (group_by_list == null) {
      return;
    }
    select_node.setGroupBy(expandAliases(group_by_list, id_to_expr));
    SqlNode having = select_node.getHaving();
    if (having == null) {
      return;
    }
    expandAliases(having, id_to_expr);
  }

  private static SqlNode expandAliases(final SqlNode node,
                                       final java.util.Map<String, SqlNode> id_to_expr) {
    if (node instanceof SqlIdentifier && id_to_expr.containsKey(node.toString())) {
      return id_to_expr.get(node.toString());
    }
    if (node instanceof SqlBasicCall) {
      SqlBasicCall node_call = (SqlBasicCall) node;
      SqlNode[] operands = node_call.getOperands();
      for (int i = 0; i < operands.length; ++i) {
        node_call.setOperand(i, expandAliases(operands[i], id_to_expr));
      }
    }
    return node;
  }

  private static SqlNodeList expandAliases(final SqlNodeList group_by_list, final java.util.Map<String, SqlNode> id_to_expr) {
    SqlNodeList new_group_by_list = new SqlNodeList(new SqlParserPos(-1, -1));
    for (SqlNode group_by : group_by_list) {
      if (!(group_by instanceof SqlIdentifier)) {
        new_group_by_list.add(group_by);
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
  protected final RelDataTypeFactory getTypeFactory() {
    if (typeFactory == null) {
      typeFactory = createTypeFactory();
    }
    return typeFactory;
  }

  protected RelDataTypeFactory createTypeFactory() {
    return new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT);
  }

  protected Prepare.CatalogReader createCatalogReader(
          RelDataTypeFactory typeFactory) {
    if (catalogReader == null) {
      catalogReader = new MapDCatalogReader(typeFactory, true).init();
    }
    return catalogReader;
  }

  protected SqlValidator createValidator(
          SqlValidatorCatalogReader catalogReader,
          RelDataTypeFactory typeFactory) {
    if (validator == null) {
      validator = new MapDTestValidator(
              getOperatorTable(),
              createCatalogReader(typeFactory),
              typeFactory,
              getConformance());
    }
    return validator;
  }

  protected SqlConformance getConformance() {
    return SqlConformance.DEFAULT;
  }

  protected final SqlOperatorTable getOperatorTable() {
    if (opTab == null) {
      opTab = createOperatorTable();
    }
    return opTab;
  }

  /**
   * Creates an operator table.
   *
   * @return New operator table
   */
  protected SqlOperatorTable createOperatorTable() {
    final MapDSqlOperatorTable tempOpTab
            = new MapDSqlOperatorTable(SqlStdOperatorTable.instance());
    // MAT 11 Nov 2015
    // Example of how to add custom function
    MapDSqlOperatorTable.addUDF(tempOpTab);
    return tempOpTab;
  }

  protected SqlToRelConverter createSqlToRelConverter(
          final SqlValidator validator,
          final Prepare.CatalogReader catalogReader,
          final RelDataTypeFactory typeFactory) {
    if (converter == null) {
      final RexBuilder rexBuilder = new RexBuilder(typeFactory);
      final RelOptCluster cluster
              = RelOptCluster.create(getPlanner(), rexBuilder);
      converter = new SqlToRelConverter(null, validator, catalogReader, cluster,
              StandardConvertletTable.INSTANCE);
    }
    return converter;
  }

  protected final RelOptPlanner getPlanner() {
    if (planner == null) {
      planner = createPlanner();
    }
    return planner;
  }

  protected RelOptPlanner createPlanner() {
    return new MapDRelOptPlanner();
  }

  protected SqlNode parseStmt(String sql) throws SqlParseException {
    return getSqlParser(sql).parseStmt();
  }

  protected SqlParser getSqlParser(String sql) {
    return SqlParser.create(sql,
            SqlParser.configBuilder()
            .setQuoting(quoting)
            .setUnquotedCasing(unquotedCasing)
            .setQuotedCasing(quotedCasing)
            .build());
  }

  private static class MapDTestValidator extends SqlValidatorImpl {

    public MapDTestValidator(
            SqlOperatorTable opTab,
            SqlValidatorCatalogReader catalogReader,
            RelDataTypeFactory typeFactory,
            SqlConformance conformance) {
      super(opTab, catalogReader, typeFactory, conformance);
    }

    // override SqlValidator
    @Override
    public boolean shouldExpandIdentifiers() {
      return true;
    }
  }
}

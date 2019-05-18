package com.mapd.parser.server.test;

import static org.junit.Assert.assertEquals;

import static java.util.Arrays.asList;

import com.mapd.calcite.parser.MapDSqlOperatorTable;

import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.prepare.MapDPlanner;
import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.SchemaVersion;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.junit.Test;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public class SqlIdentifierCapturerTest {
  private class MockSchema implements Schema {
    @Override
    public Table getTable(String name) {
      return null;
    }

    @Override
    public Set<String> getTypeNames() {
      return Collections.emptySet();
    }

    @Override
    public RelProtoDataType getType(String arg0) {
      return null;
    }

    @Override
    public Set<String> getTableNames() {
      return Collections.emptySet();
    }

    @Override
    public Collection<Function> getFunctions(String name) {
      return Collections.emptySet();
    }

    @Override
    public Set<String> getFunctionNames() {
      return Collections.emptySet();
    }

    @Override
    public Schema getSubSchema(String name) {
      return null;
    }

    @Override
    public Set<String> getSubSchemaNames() {
      return Collections.emptySet();
    }

    @Override
    public Expression getExpression(SchemaPlus parentSchema, String name) {
      return null;
    }

    @Override
    public boolean isMutable() {
      return false;
    }

    @Override
    public Schema snapshot(SchemaVersion version) {
      return null;
    }
  }

  private Planner getPlanner() {
    Schema mapd = new MockSchema() {

    };
    final SchemaPlus rootSchema = Frameworks.createRootSchema(true);
    final FrameworkConfig config =
            Frameworks.newConfigBuilder()
                    .defaultSchema(rootSchema.add("omnisci", mapd))
                    .operatorTable(
                            new MapDSqlOperatorTable(SqlStdOperatorTable.instance()))
                    .parserConfig(SqlParser.configBuilder()
                                          .setUnquotedCasing(Casing.UNCHANGED)
                                          .setCaseSensitive(false)
                                          .build())
                    .build();
    return new MapDPlanner(config);
  }

  public static String[] asArray(String... vals) {
    return vals;
  }

  public static Set<String> asSet(String... vals) {
    return new HashSet<String>(asList(vals));
  }

  public void testSelect(String sql, String[] expectedSelects) throws Exception {
    SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
    capturer.scan(getPlanner().parse(sql));

    assertEquals("selects", asSet(expectedSelects), capturer.selects);
    assertEquals("inserts", asSet(), capturer.inserts);
    assertEquals("updates", asSet(), capturer.updates);
    assertEquals("deletes", asSet(), capturer.deletes);
  }

  public void testUpdate(String sql, String[] expectedUpdates, String[] expectedSelects)
          throws Exception {
    SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
    capturer.scan(getPlanner().parse(sql));

    assertEquals("selects", asSet(expectedSelects), capturer.selects);
    assertEquals("inserts", asSet(), capturer.inserts);
    assertEquals("updates", asSet(expectedUpdates), capturer.updates);
    assertEquals("deletes", asSet(), capturer.deletes);
  }

  public void testInsert(String sql, String[] expectedInserts, String[] expectedSelects)
          throws Exception {
    SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
    capturer.scan(getPlanner().parse(sql));

    assertEquals("selects", asSet(expectedSelects), capturer.selects);
    assertEquals("inserts", asSet(expectedInserts), capturer.inserts);
    assertEquals("updates", asSet(), capturer.updates);
    assertEquals("deletes", asSet(), capturer.deletes);
  }

  public void testDelete(String sql, String[] expectedDeletes, String[] expectedSelects)
          throws Exception {
    SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
    capturer.scan(getPlanner().parse(sql));

    assertEquals("selects", asSet(expectedSelects), capturer.selects);
    assertEquals("inserts", asSet(), capturer.inserts);
    assertEquals("updates", asSet(), capturer.updates);
    assertEquals("deletes", asSet(expectedDeletes), capturer.deletes);
  }

  @Test
  public void testSelects() throws Exception {
    String sql = "SELECT * FROM sales";
    testSelect(sql, asArray("sales"));

    sql = "SELECT * FROM sales AS s";
    testSelect(sql, asArray("sales"));

    sql = "SELECT * FROM sales AS s, reports AS r WHERE s.id = r.id";
    testSelect(sql, asArray("sales", "reports"));

    sql = "SELECT * FROM sales AS s left outer join reports AS r on s.id = r.id";
    testSelect(sql, asArray("sales", "reports"));

    sql = "SELECT *, (SELECT sum(val) FROM marketing m WHERE m.id=a.id) FROM sales AS s left outer join reports AS r on s.id = r.id";
    testSelect(sql, asArray("sales", "reports", "marketing"));

    sql = "SELECT * FROM sales UNION SELECT * FROM reports UNION SELECT * FROM marketing";
    testSelect(sql, asArray("sales", "reports", "marketing"));

    sql = "SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str99') GROUP BY str HAVING n > 0 ORDER BY n DESC";
    testSelect(sql, asArray("query_rewrite_test"));

    sql = "SELECT str, SUM(y) as total_y FROM test GROUP BY str ORDER BY total_y DESC, str LIMIT 1";
    testSelect(sql, asArray("test"));

    sql = "SELECT str FROM (SELECT str, SUM(y) as total_y FROM test GROUP BY str ORDER BY total_y DESC, str LIMIT 1)";
    testSelect(sql, asArray("test"));

    sql = "SELECT deptno, dname FROM (SELECT * from dept) AS view_name LIMIT 10";
    testSelect(sql, asArray("dept"));

    sql = "WITH d1 AS (SELECT deptno, dname FROM dept LIMIT 10) SELECT ename, dname FROM emp, d1 WHERE emp.deptno = d1.deptno ORDER BY ename ASC LIMIT 10";
    testSelect(sql, asArray("emp", "dept"));
  }

  @Test
  public void testSelectsWithSchema() throws Exception {
    String sql = "SELECT * FROM mapd.sales";
    testSelect(sql, asArray("sales"));

    sql = "SELECT * FROM mapd.sales AS s";
    testSelect(sql, asArray("sales"));

    sql = "SELECT * FROM mapd.sales AS s, mapd.reports AS r WHERE s.id = r.id";
    testSelect(sql, asArray("sales", "reports"));

    sql = "SELECT * FROM mapd.sales AS s left outer join mapd.reports AS r on s.id = r.id";
    testSelect(sql, asArray("sales", "reports"));

    sql = "SELECT *, (SELECT sum(val) FROM mapd.marketing m WHERE m.id=a.id) FROM mapd.sales AS s left outer join mapd.reports AS r on s.id = r.id";
    testSelect(sql, asArray("sales", "reports", "marketing"));

    sql = "SELECT * FROM mapd.sales UNION SELECT * FROM mapd.reports UNION SELECT * FROM mapd.marketing";
    testSelect(sql, asArray("sales", "reports", "marketing"));

    sql = "SELECT COUNT(*) AS n, str FROM mapd.query_rewrite_test WHERE str IN ('str2', 'str99') GROUP BY str HAVING n > 0 ORDER BY n DESC";
    testSelect(sql, asArray("query_rewrite_test"));

    sql = "SELECT str, SUM(y) as total_y FROM mapd.test GROUP BY str ORDER BY total_y DESC, str LIMIT 1";
    testSelect(sql, asArray("test"));

    sql = "SELECT str FROM (SELECT str, SUM(y) as total_y FROM mapd.test GROUP BY str ORDER BY total_y DESC, str LIMIT 1)";
    testSelect(sql, asArray("test"));

    sql = "SELECT deptno, dname FROM (SELECT * from mapd.dept) AS view_name LIMIT 10";
    testSelect(sql, asArray("dept"));

    sql = "WITH d1 AS (SELECT deptno, dname FROM mapd.dept LIMIT 10) SELECT ename, dname FROM mapd.emp, d1 WHERE emp.deptno = d1.deptno ORDER BY ename ASC LIMIT 10";
    testSelect(sql, asArray("emp", "dept"));
  }

  @Test
  public void testInserts() throws Exception {
    String sql = "INSERT INTO sales VALUES(10)";
    testInsert(sql, asArray("sales"), asArray());

    sql = "INSERT INTO sales(id, target) VALUES(10, 21321)";
    testInsert(sql, asArray("sales"), asArray());

    sql = "INSERT INTO sales(id, target) VALUES(10, (SELECT max(r.val) FROM reports AS r))";
    testInsert(sql, asArray("sales"), asArray("reports"));

    sql = "INSERT INTO sales(id, target) VALUES((SELECT m.id FROM marketing m), (SELECT max(r.val) FROM reports AS r))";
    testInsert(sql, asArray("sales"), asArray("reports", "marketing"));
  }

  @Test
  public void testInsertsWithSchema() throws Exception {
    String sql = "INSERT INTO mapd.sales VALUES(10)";
    testInsert(sql, asArray("sales"), asArray());

    sql = "INSERT INTO mapd.sales(id, target) VALUES(10, 21321)";
    testInsert(sql, asArray("sales"), asArray());

    sql = "INSERT INTO mapd.sales(id, target) VALUES(10, (SELECT max(r.val) FROM mapd.reports AS r))";
    testInsert(sql, asArray("sales"), asArray("reports"));

    sql = "INSERT INTO mapd.sales(id, target) VALUES((SELECT m.id FROM mapd.marketing m), (SELECT max(r.val) FROM mapd.reports AS r))";
    testInsert(sql, asArray("sales"), asArray("reports", "marketing"));
  }

  @Test
  public void testUpdates() throws Exception {
    String sql = "UPDATE sales SET id=10";
    testUpdate(sql, asArray("sales"), asArray());

    sql = "UPDATE sales SET id=10 WHERE id=1";
    testUpdate(sql, asArray("sales"), asArray());

    sql = "UPDATE sales SET id=(SELECT max(r.val) FROM reports AS r)";
    testUpdate(sql, asArray("sales"), asArray("reports"));

    sql = "UPDATE sales SET id=(SELECT max(r.val) FROM reports AS r) WHERE id=(SELECT max(m.val) FROM marketing AS m)";
    testUpdate(sql, asArray("sales"), asArray("reports", "marketing"));

    sql = "UPDATE shardkey SET y=99 WHERE x=(SELECT max(id) from v2 LIMIT 1)";
    testUpdate(sql, asArray("shardkey"), asArray("v2"));
  }

  @Test
  public void testUpdatesWithSchema() throws Exception {
    String sql = "UPDATE mapd.sales SET id=10";
    testUpdate(sql, asArray("sales"), asArray());

    sql = "UPDATE mapd.sales SET id=10 WHERE id=1";
    testUpdate(sql, asArray("sales"), asArray());

    sql = "UPDATE mapd.sales SET id=(SELECT max(r.val) FROM mapd.reports AS r)";
    testUpdate(sql, asArray("sales"), asArray("reports"));

    sql = "UPDATE mapd.sales SET id=(SELECT max(r.val) FROM mapd.reports AS r) WHERE id=(SELECT max(m.val) FROM mapd.marketing AS m)";
    testUpdate(sql, asArray("sales"), asArray("reports", "marketing"));

    sql = "UPDATE mapd.shardkey SET y=99 WHERE x=(SELECT max(id) from mapd.v2 LIMIT 1)";
    testUpdate(sql, asArray("shardkey"), asArray("v2"));
  }

  @Test
  public void testDeletes() throws Exception {
    String sql = "DELETE FROM sales";
    testDelete(sql, asArray("sales"), asArray());

    sql = "DELETE FROM sales WHERE id=1";
    testDelete(sql, asArray("sales"), asArray());

    sql = "DELETE FROM sales WHERE id=(SELECT max(r.val) FROM reports AS r)";
    testDelete(sql, asArray("sales"), asArray("reports"));

    sql = "DELETE FROM sales WHERE id=(SELECT max(r.val) FROM reports AS r) AND id=(SELECT max(m.val) FROM marketing AS m)";
    testDelete(sql, asArray("sales"), asArray("reports", "marketing"));
  }

  @Test
  public void testDeletesWithSchema() throws Exception {
    String sql = "DELETE FROM mapd.sales";
    testDelete(sql, asArray("sales"), asArray());

    sql = "DELETE FROM mapd.sales WHERE id=1";
    testDelete(sql, asArray("sales"), asArray());

    sql = "DELETE FROM mapd.sales WHERE id=(SELECT max(r.val) FROM mapd.reports AS r)";
    testDelete(sql, asArray("sales"), asArray("reports"));

    sql = "DELETE FROM mapd.sales WHERE id=(SELECT max(r.val) FROM mapd.reports AS r) AND id=(SELECT max(m.val) FROM mapd.marketing AS m)";
    testDelete(sql, asArray("sales"), asArray("reports", "marketing"));
  }
}

package com.mapd.parser.server.test;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;

import java.util.HashSet;
import java.util.Set;

import org.apache.calcite.prepare.SqlIdentifierCapturer;
import org.apache.calcite.sql.parser.SqlParser;
import org.junit.Test;

public class SqlIdentifierCapturerTest {
	public static String[] asArray(String... vals) {
		return vals;
	}

	public static Set<String> asSet(String... vals) {
		return new HashSet<String>(asList(vals));
	}

	public void testSelect(String sql, String[] expectedSelects) throws Exception {
		SqlParser parser = SqlParser.create(sql);

		SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
		capturer.scan(parser.parseQuery());

		assertEquals("selects", asSet(expectedSelects), capturer.selects);
		assertEquals("inserts", asSet(), capturer.inserts);
		assertEquals("updates", asSet(), capturer.updates);
		assertEquals("deletes", asSet(), capturer.deletes);
	}

	public void testUpdate(String sql, String[] expectedUpdates, String[] expectedSelects) throws Exception {
		SqlParser parser = SqlParser.create(sql);

		SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
		capturer.scan(parser.parseQuery());

		assertEquals("selects", asSet(expectedSelects), capturer.selects);
		assertEquals("inserts", asSet(), capturer.inserts);
		assertEquals("updates", asSet(expectedUpdates), capturer.updates);
		assertEquals("deletes", asSet(), capturer.deletes);
	}

	public void testInsert(String sql, String[] expectedInserts, String[] expectedSelects) throws Exception {
		SqlParser parser = SqlParser.create(sql);

		SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
		capturer.scan(parser.parseQuery());

		assertEquals("selects", asSet(expectedSelects), capturer.selects);
		assertEquals("inserts", asSet(expectedInserts), capturer.inserts);
		assertEquals("updates", asSet(), capturer.updates);
		assertEquals("deletes", asSet(), capturer.deletes);
	}

	public void testDelete(String sql, String[] expectedDeletes, String[] expectedSelects) throws Exception {
		SqlParser parser = SqlParser.create(sql);

		SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
		capturer.scan(parser.parseQuery());

		assertEquals("selects", asSet(expectedSelects), capturer.selects);
		assertEquals("inserts", asSet(), capturer.inserts);
		assertEquals("updates", asSet(), capturer.updates);
		assertEquals("deletes", asSet(expectedDeletes), capturer.deletes);
	}

	@Test
	public void testSelects() throws Exception {
		String sql = "SELECT * FROM sales";
		testSelect(sql, asArray("SALES"));

		sql = "SELECT * FROM sales AS s";
		testSelect(sql, asArray("SALES"));

		sql = "SELECT * FROM sales AS s, reports AS r WHERE s.id = r.id";
		testSelect(sql, asArray("SALES", "REPORTS"));

		sql = "SELECT * FROM sales AS s left outer join reports AS r on s.id = r.id";
		testSelect(sql, asArray("SALES", "REPORTS"));

		sql = "SELECT *, (SELECT sum(val) FROM marketing m WHERE m.id=a.id) FROM sales AS s left outer join reports AS r on s.id = r.id";
		testSelect(sql, asArray("SALES", "REPORTS", "MARKETING"));

		sql = "SELECT * FROM sales UNION SELECT * FROM reports UNION SELECT * FROM marketing";
		testSelect(sql, asArray("SALES", "REPORTS", "MARKETING"));
	}

	@Test
	public void testInserts() throws Exception {
		String sql = "INSERT INTO sales VALUES(10)";
		testInsert(sql, asArray("SALES"), asArray());

		sql = "INSERT INTO sales(id, target) VALUES(10, 21321)";
		testInsert(sql, asArray("SALES"), asArray());

		sql = "INSERT INTO sales(id, target) VALUES(10, (SELECT max(r.val) FROM reports AS r))";
		testInsert(sql, asArray("SALES"), asArray("REPORTS"));

		sql = "INSERT INTO sales(id, target) VALUES((SELECT m.id FROM marketing m), (SELECT max(r.val) FROM reports AS r))";
		testInsert(sql, asArray("SALES"), asArray("REPORTS", "MARKETING"));
	}

	@Test
	public void testUpdates() throws Exception {
		String sql = "UPDATE sales SET id=10";
		testUpdate(sql, asArray("SALES"), asArray());

		sql = "UPDATE sales SET id=10 WHERE id=1";
		testUpdate(sql, asArray("SALES"), asArray());

		sql = "UPDATE sales SET id=(SELECT max(r.val) FROM reports AS r)";
		testUpdate(sql, asArray("SALES"), asArray("REPORTS"));

		sql = "UPDATE sales SET id=(SELECT max(r.val) FROM reports AS r) WHERE id=(SELECT max(m.val) FROM marketing AS m)";
		testUpdate(sql, asArray("SALES"), asArray("REPORTS", "MARKETING"));
	}

	@Test
	public void testDeletes() throws Exception {
		String sql = "DELETE FROM sales";
		testDelete(sql, asArray("SALES"), asArray());

		sql = "DELETE FROM sales WHERE id=1";
		testDelete(sql, asArray("SALES"), asArray());

		sql = "DELETE FROM sales WHERE id=(SELECT max(r.val) FROM reports AS r)";
		testDelete(sql, asArray("SALES"), asArray("REPORTS"));

		sql = "DELETE FROM sales WHERE id=(SELECT max(r.val) FROM reports AS r) AND id=(SELECT max(m.val) FROM marketing AS m)";
		testDelete(sql, asArray("SALES"), asArray("REPORTS", "MARKETING"));
	}
}

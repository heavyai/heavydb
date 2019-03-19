package org.apache.calcite.prepare;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Collection;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlDelete;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlInsert;
import org.apache.calcite.sql.SqlJoin;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.SqlUpdate;
import org.apache.calcite.sql.SqlWith;
import org.apache.calcite.sql.SqlWithItem;
import org.apache.calcite.sql.parser.SqlParser;

import com.mapd.calcite.parser.MapDSchema;

/**
 * scans from a root {@link SqlNode} and retrieves all {@link SqlIdentifier}s
 * used in a query.
 */
public class SqlIdentifierCapturer {
  private static final Map<Class<?>, Set<Method>> GETTERS_CACHE =
          new ConcurrentHashMap<>();

  private IdentityHashMap<SqlNode, SqlNode> visitedNodes = new IdentityHashMap<>();

  private Stack<Set<String>> currentList = new Stack<>();

  public final Set<String> selects = new HashSet<>();
  public final Set<String> inserts = new HashSet<>();
  public final Set<String> updates = new HashSet<>();
  public final Set<String> deletes = new HashSet<>();

  private final Set<String> ignore = new HashSet<>();

  { currentList.push(ignore); }

  public void scan(SqlNode root) {
    if (null == root) {
      return;
    }

    if (root instanceof SqlLiteral || root instanceof SqlDataTypeSpec) {
      return;
    }

    if (null != visitedNodes.put(root, root)) {
      return;
    }

    if (root instanceof SqlNodeList) {
      SqlNodeList snl = (SqlNodeList) root;
      for (SqlNode node : snl) {
        scan(node);
      }
      return;
    }

    if (root instanceof SqlIdentifier) {
      // only the last element!
      currentList.peek().add(((SqlIdentifier) root).names.reverse().get(0));
      return;
    }

    if (root instanceof SqlBasicCall) {
      SqlBasicCall call = (SqlBasicCall) root;
      if (call.getOperator().getKind() == SqlKind.AS) {
        // only really interested in the first operand
        scan(call.getOperands()[0]);
        return;
      }
    }

    if (root instanceof SqlOrderBy) {
      scan(((SqlOrderBy) root).fetch);
      scan(((SqlOrderBy) root).offset);
      scan(((SqlOrderBy) root).query);
      return;
    }

    boolean needsPop = false;
    if (root instanceof SqlSelect) {
      currentList.push(selects);
      scan(((SqlSelect) root).getFrom());
      currentList.pop();
      currentList.push(ignore);
      needsPop = true;
    } else if (root instanceof SqlInsert) {
      currentList.push(inserts);
      scan(((SqlInsert) root).getTargetTable());
      currentList.pop();
      currentList.push(ignore);
      needsPop = true;
    } else if (root instanceof SqlUpdate) {
      currentList.push(updates);
      scan(((SqlUpdate) root).getTargetTable());
      currentList.pop();
      currentList.push(ignore);
      needsPop = true;
    } else if (root instanceof SqlDelete) {
      currentList.push(deletes);
      scan(((SqlDelete) root).getTargetTable());
      currentList.pop();
      currentList.push(ignore);
      needsPop = true;
    } else if (root instanceof SqlJoin) {
      currentList.push(ignore);
      scan(((SqlJoin) root).getCondition());
      currentList.pop();
    }

    Set<Method> methods = getRelevantGetters(root);
    for (Method m : methods) {
      Object value = null;
      try {
        value = m.invoke(root);
      } catch (Exception e) {
      }

      if (value instanceof SqlNode[]) {
        SqlNode[] nodes = (SqlNode[]) value;
        for (SqlNode node : nodes) {
          scan(node);
        }
      } else if (value instanceof SqlNode) {
        scan((SqlNode) value);
      } else if (value instanceof Collection) {
        for (Object vobj : ((Collection<?>) value)) {
          if (vobj instanceof SqlNode) {
            scan((SqlNode) vobj);
          }
        }
      }
    }

    if (root instanceof SqlWith) {
      SqlWith with = (SqlWith) root;

      for (SqlNode node : with.withList) {
        SqlWithItem item = (SqlWithItem) node;
        selects.remove(item.name.getSimple());
      }
    }

    if (needsPop) {
      currentList.pop();
    }
  }

  Set<Method> getRelevantGetters(Object obj) {
    Class<?> root = obj.getClass();

    Set<Method> methods = GETTERS_CACHE.get(root);
    if (null != methods) {
      return methods;
    } else {
      methods = new HashSet<>();
    }

    while (root != null) {
      if (root == SqlNode.class) break;

      for (Method m : root.getDeclaredMethods()) {
        if (m.getParameterTypes().length > 0) continue;

        if (!Modifier.isPublic(m.getModifiers())) continue;

        Class<?> returnType = m.getReturnType();
        if (!SqlNode.class.isAssignableFrom(returnType) && SqlNode[].class != returnType
                && !Collection.class.isAssignableFrom(returnType)) {
          continue;
        }

        methods.add(m);
      }

      root = root.getSuperclass();
    }

    GETTERS_CACHE.put(obj.getClass(), methods);

    return methods;
  }

  public String toString() {
    String out = "";
    out += " Selects: " + selects + "\n";
    out += " Inserts: " + inserts + "\n";
    out += " Updates: " + updates + "\n";
    out += " Deletes: " + deletes + "\n";
    out += " Ignore : " + ignore + "\n";

    return out;
  }

  public static void main(String[] args) throws Exception {
    String sql = "UPDATE sales set f=(SELECT max(r.num) from report as r)";
    sql = "INSER INTO sales (a, b, c ) VALUES(10, (SELECT max(foo) from bob), 0)";
    sql = "SELECT * FROM sales a left outer join (select (select max(id) from rupert) from report2) r on a.id=(select max(r.di) from test)";

    SqlParser parser = SqlParser.create(sql);

    SqlIdentifierCapturer capturer = new SqlIdentifierCapturer();
    capturer.scan(parser.parseQuery());

    System.out.println(capturer.selects);
    System.out.println(capturer.inserts);
    System.out.println(capturer.updates);
    System.out.println(capturer.deletes);
    System.out.println(capturer.ignore);
  }
}
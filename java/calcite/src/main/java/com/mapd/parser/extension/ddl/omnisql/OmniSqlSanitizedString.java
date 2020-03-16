package com.mapd.parser.extension.ddl.omnisql;

import org.apache.calcite.sql.SqlNode;

public class OmniSqlSanitizedString {
  private String string;

  public OmniSqlSanitizedString(final String s) {
    this.string = sanitizeString(s);
  }

  public OmniSqlSanitizedString(final SqlNode node) {
    this.string = sanitizeString(node.toString());
  }

  private String sanitizeString(final String s) {
    if (s.startsWith("'") && s.endsWith("'")) {
      return s.substring(1, s.length() - 1);
    }
    return s;
  }

  @Override
  public String toString() {
    return string;
  }
}

package com.mapd.parser.extension.ddl.heavysql;

import org.apache.calcite.sql.SqlNode;

public class HeavySqlSanitizedString {
  private String string;

  public HeavySqlSanitizedString(final String s) {
    this.string = sanitizeString(s);
  }

  public HeavySqlSanitizedString(final SqlNode node) {
    this.string = sanitizeString(node.toString());
  }

  private String sanitizeString(final String s) {
    String sanitized_s = s;
    if (s.startsWith("'") && s.endsWith("'")) {
      sanitized_s = s.substring(1, s.length() - 1);
    }
    sanitized_s = sanitized_s.replace("''", "'");
    return sanitized_s;
  }

  @Override
  public String toString() {
    return string;
  }
}

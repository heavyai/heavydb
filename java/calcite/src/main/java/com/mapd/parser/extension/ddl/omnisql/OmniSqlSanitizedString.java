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

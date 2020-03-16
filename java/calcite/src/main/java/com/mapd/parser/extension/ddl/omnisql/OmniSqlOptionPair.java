package com.mapd.parser.extension.ddl.omnisql;

import org.apache.calcite.util.Pair;

public class OmniSqlOptionPair extends Pair<String, OmniSqlSanitizedString> {
  public OmniSqlOptionPair(String option, OmniSqlSanitizedString value) {
    super(option, value);
  }
}

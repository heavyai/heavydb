package com.mapd.parser.extension.ddl.heavysql;

import com.mapd.parser.extension.ddl.heavysql.HeavySqlSanitizedString;

import org.apache.calcite.util.Pair;

public class HeavySqlOptionPair extends Pair<String, HeavySqlSanitizedString> {
  public HeavySqlOptionPair(String option, HeavySqlSanitizedString value) {
    super(option, value);
  }
}

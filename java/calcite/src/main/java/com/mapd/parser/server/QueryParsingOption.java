package com.mapd.parser.server;

public class QueryParsingOption {
  public boolean legacy_syntax;
  public boolean is_explain;
  public boolean check_privileges;

  
  public QueryParsingOption() {
    this.legacy_syntax = false;
    this.is_explain = false;
    this.check_privileges = false;
  }
  
  public QueryParsingOption(boolean legacy_syntax,
          boolean is_explain,
          boolean check_privileges) {
    this.legacy_syntax = legacy_syntax;
    this.is_explain = is_explain;
    this.check_privileges = check_privileges;
  }
}

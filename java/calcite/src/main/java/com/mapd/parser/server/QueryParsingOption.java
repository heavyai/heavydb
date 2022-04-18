package com.mapd.parser.server;

public class QueryParsingOption {
  public boolean legacySyntax;
  public boolean isExplain;
  public boolean checkPrivileges;

  
  public QueryParsingOption() {
    this.legacySyntax = false;
    this.isExplain = false;
    this.checkPrivileges = false;
  }
  
  public QueryParsingOption(boolean legacySyntax,
          boolean isExplain,
          boolean checkPrivileges) {
    this.legacySyntax = legacySyntax;
    this.isExplain = isExplain;
    this.checkPrivileges = checkPrivileges;
  }
}

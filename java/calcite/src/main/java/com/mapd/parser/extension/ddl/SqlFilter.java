package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

import java.util.Arrays;
public class SqlFilter implements JsonSerializableDdl {
  public enum Operation { EQUALS, LIKE }
  public enum Chain { AND, OR }
  @Expose
  private String attribute;
  @Expose
  private String value;
  @Expose
  private Operation operation;
  @Expose
  private Chain chain;

  public SqlFilter(final String attribute,
          final String value,
          final Operation operation,
          final Chain chain) {
    this.attribute = attribute;
    this.value = value;
    this.operation = operation;
    this.chain = chain;
  }
}

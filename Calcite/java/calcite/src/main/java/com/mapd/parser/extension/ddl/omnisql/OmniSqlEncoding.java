package com.mapd.parser.extension.ddl.omnisql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;

public class OmniSqlEncoding extends OmniSqlJson {
  @Expose
  private String type;
  @Expose
  private Integer size;

  public OmniSqlEncoding(String type, Integer size) {
    requireNonNull(type);
    this.type = type;
    this.size = size;
  }
}

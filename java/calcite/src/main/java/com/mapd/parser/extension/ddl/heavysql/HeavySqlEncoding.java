package com.mapd.parser.extension.ddl.heavysql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;

public class HeavySqlEncoding extends HeavySqlJson {
  @Expose
  private String type;
  @Expose
  private Integer size;

  public HeavySqlEncoding(String type, Integer size) {
    requireNonNull(type);
    this.type = type;
    this.size = size;
  }
}

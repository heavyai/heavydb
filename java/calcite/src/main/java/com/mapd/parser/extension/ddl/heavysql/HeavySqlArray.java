package com.mapd.parser.extension.ddl.heavysql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;
import com.mapd.parser.extension.ddl.heavysql.*;

import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlIdentifier;

public class HeavySqlArray extends HeavySqlJson {
  @Expose
  private String elementType;
  @Expose
  private Integer size;

  public HeavySqlArray(final String elementType, final Integer size) {
    this.elementType = elementType;
    this.size = size;
  }

  Integer getSize() {
    return size;
  }
}

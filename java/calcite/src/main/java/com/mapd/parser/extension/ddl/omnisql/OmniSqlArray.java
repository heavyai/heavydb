package com.mapd.parser.extension.ddl.omnisql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlIdentifier;

public class OmniSqlArray extends OmniSqlJson {
  @Expose
  private String elementType;
  @Expose
  private Integer size;

  public OmniSqlArray(final String elementType, final Integer size) {
    this.elementType = elementType;
    this.size = size;
  }

  Integer getSize() {
    return size;
  }
}

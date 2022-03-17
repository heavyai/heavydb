package com.mapd.parser.extension.ddl.heavysql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;
import com.mapd.parser.extension.ddl.heavysql.*;

import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlIdentifier;

public class HeavySqlColumn extends HeavySqlJson {
  @Expose
  private String name;
  @Expose
  private HeavySqlDataType dataType;
  @Expose
  private HeavySqlOptionsMap options;

  public HeavySqlColumn(final SqlIdentifier name,
          final HeavySqlDataType type,
          final HeavySqlEncoding encoding,
          final HeavySqlOptionsMap options) {
    requireNonNull(name);
    this.name = name.toString();
    this.dataType = type;
    this.options = options;
  }
} // HeavySqlColumn.

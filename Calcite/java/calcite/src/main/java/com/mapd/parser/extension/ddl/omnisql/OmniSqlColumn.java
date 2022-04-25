package com.mapd.parser.extension.ddl.omnisql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlIdentifier;

public class OmniSqlColumn extends OmniSqlJson {
  @Expose
  private String name;
  @Expose
  private OmniSqlDataType dataType;
  @Expose
  private OmniSqlOptionsMap options;

  public OmniSqlColumn(final SqlIdentifier name,
          final OmniSqlDataType type,
          final OmniSqlEncoding encoding,
          final OmniSqlOptionsMap options) {
    requireNonNull(name);
    this.name = name.toString();
    this.dataType = type;
    this.options = options;
  }
} // OmniSqlColumn.

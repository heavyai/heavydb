package com.mapd.parser.extension.ddl.omnisql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;
import com.mapd.parser.extension.ddl.JsonSerializableDdl;
import com.mapd.parser.extension.ddl.omnisql.*;

import org.apache.calcite.sql.SqlDataTypeSpec;

public class OmniSqlDataType extends OmniSqlJson {
  @Expose
  private String type;
  @Expose
  private OmniSqlArray array;
  @Expose
  private String precision;
  @Expose
  private String scale;
  @Expose
  private String notNull;
  @Expose
  private String coordinateSystem;
  @Expose
  private OmniSqlEncoding encoding;

  public OmniSqlDataType(final SqlDataTypeSpec type,
          final boolean notNull,
          final OmniSqlArray array,
          final Integer precision,
          final Integer scale,
          final Integer coordinateSystem,
          final OmniSqlEncoding encoding) {
    requireNonNull(type);
    this.type = type.toString();
    this.array = array;
    this.precision = (precision == null) ? null : precision.toString();
    this.scale = (scale == null) ? null : scale.toString();
    this.notNull = notNull ? "true" : "false";
    this.coordinateSystem =
            (coordinateSystem == null) ? null : coordinateSystem.toString();
    this.encoding = encoding;
  }
}

package com.mapd.parser.extension.ddl.omnisql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlDataTypeSpec;

public class OmniSqlDataType extends OmniSqlJson {
  @Expose
  private String type;
  @Expose
  private OmniSqlArray array;
  @Expose
  private Integer precision;
  @Expose
  private Integer scale;
  @Expose
  private boolean notNull;
  @Expose
  private Integer coordinateSystem;
  @Expose
  private OmniSqlEncoding encoding;

  public OmniSqlDataType(final SqlDataTypeSpec type,
          final boolean notNull,
          final OmniSqlArray array,
          final OmniSqlEncoding encoding) {
    requireNonNull(type);
    if (type.getTypeNameSpec() instanceof OmniSqlTypeNameSpec) {
      OmniSqlTypeNameSpec omniSqlTypeNameSpec =
              (OmniSqlTypeNameSpec) type.getTypeNameSpec();
      this.type = omniSqlTypeNameSpec.getName();
      this.coordinateSystem = omniSqlTypeNameSpec.getCoordinate();
    } else {
      this.type = type.getTypeName().toString();
    }
    if (type.getTypeNameSpec() instanceof SqlBasicTypeNameSpec) {
      SqlBasicTypeNameSpec typeNameSpec = (SqlBasicTypeNameSpec) type.getTypeNameSpec();
      this.precision =
              typeNameSpec.getPrecision() == -1 ? null : typeNameSpec.getPrecision();
      this.scale = typeNameSpec.getScale() == -1 ? null : typeNameSpec.getScale();
    }
    if (array != null) {
      this.array = new OmniSqlArray(this.type, array.getSize());
      this.type = "ARRAY";
    }
    this.notNull = notNull;
    this.encoding = encoding;
  }
}

package com.mapd.parser.extension.ddl.heavysql;

import static java.util.Objects.requireNonNull;

import com.google.gson.annotations.Expose;

import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlDataTypeSpec;

public class HeavySqlDataType extends HeavySqlJson {
  @Expose
  private String type;
  @Expose
  private HeavySqlArray array;
  @Expose
  private Integer precision;
  @Expose
  private Integer scale;
  @Expose
  private boolean notNull;
  @Expose
  private Integer coordinateSystem;
  @Expose
  private HeavySqlEncoding encoding;

  public HeavySqlDataType(final SqlDataTypeSpec type,
          final boolean notNull,
          final HeavySqlArray array,
          final HeavySqlEncoding encoding) {
    requireNonNull(type);
    if (type.getTypeNameSpec() instanceof HeavySqlTypeNameSpec) {
      HeavySqlTypeNameSpec typeNameSpec = (HeavySqlTypeNameSpec) type.getTypeNameSpec();
      this.type = typeNameSpec.getName();
      this.coordinateSystem = typeNameSpec.getCoordinate();
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
      this.array = new HeavySqlArray(this.type, array.getSize());
      this.type = "ARRAY";
    }
    this.notNull = notNull;
    this.encoding = encoding;
  }
}

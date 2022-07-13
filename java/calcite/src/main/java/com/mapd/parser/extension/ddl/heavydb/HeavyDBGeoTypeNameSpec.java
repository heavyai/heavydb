package com.mapd.parser.extension.ddl.heavydb;

import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.EscapedStringJsonBuilder;
import org.apache.calcite.util.Pair;

import java.util.Map;

public class HeavyDBGeoTypeNameSpec extends SqlBasicTypeNameSpec {
  private HeavyDBEncoding encodingType;
  private Integer encodingSize;
  private HeavyDBGeo geoType;
  private boolean isGeography;
  private Integer coordinateSystem;

  public HeavyDBGeoTypeNameSpec(HeavyDBGeo geoType,
          Integer coordinateSystem,
          boolean isGeography,
          Pair<HeavyDBEncoding, Integer> encodingInfo,
          SqlParserPos pos) {
    super(SqlTypeName.GEOMETRY, pos);
    this.geoType = geoType;
    this.coordinateSystem = coordinateSystem;
    if (encodingInfo != null) {
      this.encodingType = encodingInfo.left;
      this.encodingSize = encodingInfo.right;
    }
  }

  public Map<String, Object> toJsonMap(Map<String, Object> map) {
    EscapedStringJsonBuilder jsonBuilder = new EscapedStringJsonBuilder();
    if (isGeography) {
      jsonBuilder.put(map, "sqltype", "GEOGRAPHY");
    } else {
      jsonBuilder.put(map, "sqltype", "GEOMETRY");
    }

    if (encodingType != null) {
      jsonBuilder.put(map, "encodingType", encodingType.name());
      jsonBuilder.put(map, "encodingSize", encodingSize);
    }

    jsonBuilder.put(map, "subtype", geoType.toString());
    jsonBuilder.put(map, "coordinateSystem", coordinateSystem);

    return map;
  }
}

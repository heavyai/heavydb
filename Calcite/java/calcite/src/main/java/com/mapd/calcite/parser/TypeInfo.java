package com.mapd.calcite.parser;

public class TypeInfo {
  public enum DeviceType { CPU, GPU }

  public enum DatumType {
    SMALLINT,
    INT,
    BIGINT,
    FLOAT,
    DECIMAL,
    DOUBLE,
    STR,
    TIME,
    TIMESTAMP,
    DATE,
    BOOL,
    INTERVAL_DAY_TIME,
    INTERVAL_YEAR_MONTH,
    POINT,
    LINESTRING,
    POLYGON,
    MULTIPOLYGON,
    TINYINT,
    GEOMETRY,
    GEOGRAPHY
  }

  public enum EncodingType { NONE, FIXED, RL, DIFF, DICT, SPARSE, GEOINT, DATE_IN_DAYS }

  public DatumType type;
  public EncodingType encoding;
  public boolean nullable;
  public boolean isArray;
  public int precision;
  public int scale;
}

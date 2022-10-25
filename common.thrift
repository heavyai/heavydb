namespace java ai.heavy.thrift.server
namespace py heavydb.common

enum TDeviceType {
  CPU,
  GPU
}

enum TDatumType {
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
  GEOGRAPHY,
  MULTILINESTRING,
  MULTIPOINT
}

enum TEncodingType {
  NONE,
  FIXED,
  RL,
  DIFF,
  DICT,
  SPARSE,
  GEOINT,
  DATE_IN_DAYS,
  ARRAY,
  ARRAY_DICT
}

struct TStringDictKey {
  1: i32 db_id;
  2: i32 dict_id;
}

struct TTypeInfo {
  1: TDatumType type;
  4: TEncodingType encoding;
  2: bool nullable;
  3: bool is_array;
  5: i32 precision;
  6: i32 scale;
  7: i32 comp_param;
  8: optional i32 size=-1;
  9: optional TStringDictKey dict_key;
}


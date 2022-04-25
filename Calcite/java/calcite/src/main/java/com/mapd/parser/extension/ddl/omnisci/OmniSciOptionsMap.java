package com.mapd.parser.extension.ddl.omnisci;

import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.type.SqlTypeName;

import java.util.HashMap;

public class OmniSciOptionsMap extends HashMap<String, Object> {
  static public void add(OmniSciOptionsMap map, String key, SqlNode value) {
    if (value instanceof SqlLiteral) {
      SqlLiteral literalValue = (SqlLiteral) value;
      if (SqlTypeName.STRING_TYPES.contains(literalValue.getTypeName())) {
        map.put(key, ((SqlLiteral) value).getValueAs(String.class));
      } else {
        map.put(key, ((SqlLiteral) value).getValue());
      }
    } else {
      throw new CalciteException("Unsupported with value type for value : `"
                      + value.toString() + "` (option : `" + key + "`)",
              null);
    }
  }
}

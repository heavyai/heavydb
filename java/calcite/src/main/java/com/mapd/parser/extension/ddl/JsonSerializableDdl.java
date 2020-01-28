package com.mapd.parser.extension.ddl;

import com.mapd.calcite.parser.MapDSerializer;

public interface JsonSerializableDdl {
  default String toJsonString() {
    return MapDSerializer.toJsonString(this);
  }
}

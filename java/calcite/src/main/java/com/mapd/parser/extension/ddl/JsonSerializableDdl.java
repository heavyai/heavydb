package com.mapd.parser.extension.ddl;

import com.mapd.calcite.parser.HeavyDBSerializer;

public interface JsonSerializableDdl {
  default String toJsonString() {
    return HeavyDBSerializer.toJsonString(this);
  }
}

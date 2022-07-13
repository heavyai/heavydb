package com.mapd.parser.extension.ddl.heavysql;

import com.mapd.parser.extension.ddl.JsonSerializableDdl;

public class HeavySqlJson implements JsonSerializableDdl {
  @Override
  public String toString() {
    return toJsonString();
  }
}

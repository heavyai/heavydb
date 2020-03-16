package com.mapd.parser.extension.ddl.omnisql;

import com.mapd.parser.extension.ddl.JsonSerializableDdl;

public class OmniSqlJson implements JsonSerializableDdl {
  @Override
  public String toString() {
    return toJsonString();
  }
}

package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

public class DdlResponse {
  @Expose
  private JsonSerializableDdl payload;
  @Expose
  private final String statementType = "DDL";

  public void setPayload(final JsonSerializableDdl payload) {
    this.payload = payload;
  }
}

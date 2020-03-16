package com.mapd.parser.extension.ddl;

import com.google.gson.annotations.Expose;

public class DdlResponse {
  @Expose
  private final String statementType = "DDL";
  @Expose
  private JsonSerializableDdl payload;

  public void setPayload(final JsonSerializableDdl payload) {
    this.payload = payload;
  }
}

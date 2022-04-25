package com.mapd.parser.server;

public class InvalidParseRequest extends Exception {
  public int code;
  public String msg;

  public InvalidParseRequest(int code, String msg) {
    this.code = code;
    this.msg = msg;
  }
}

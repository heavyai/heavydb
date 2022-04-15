package com.mapd.parser.server;

public class InvalidParseRequest extends Exception {
  public int code;
  public String msg;

  public InvalidParseRequest(int code_, String msg_) {
    this.code = code_;
    this.msg = msg_;
  }
}

/*
 *  Some cool MapD License
 */
package com.mapd.parser.server;

/**
 *
 * @author michael
 */
public class CalciteReturn {
  private final String returnText;
  private final long elapsedTime;
  private final boolean failed;

  CalciteReturn(String string, long l, boolean b) {
    returnText = string;
    elapsedTime= l;
    failed = b;
  }

  public String getText() {
    return returnText;
  }

  public long getElapsedTime(){
    return elapsedTime;
  }

  public boolean hasFailed(){
    return failed;
  }
}

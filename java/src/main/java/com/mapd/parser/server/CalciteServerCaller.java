/*
 * Some cool MapD License
 */
package com.mapd.parser.server;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CalciteServerCaller {

  private final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerCaller.class);

  public static void main(String[] args) {
    CalciteServerWrapper calciteServerWrapper = null;
    switch (args.length) {
      case 0:
        calciteServerWrapper = new CalciteServerWrapper();
        break;
      case 2:
        calciteServerWrapper = new CalciteServerWrapper(Integer.valueOf(args[0]), Integer.valueOf(args[1]));
        break;
      default:
        MAPDLOGGER.error("Incorrect parameters; CalciteServerCaller [calcitePort, mapDPort]");
        System.exit(1);
    }
    while (true) {
      try {
        Thread t = new Thread(calciteServerWrapper);
        t.start();
        t.join();
        if (calciteServerWrapper.shutdown()) {
          break;
        }
      } catch (Exception x) {
        x.printStackTrace();
      }
    }
  }
}
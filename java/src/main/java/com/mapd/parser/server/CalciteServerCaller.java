/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.parser.server;

import com.mapd.thrift.calciteserver.CalciteServer;
import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TServer.Args;
import org.apache.thrift.server.TSimpleServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CalciteServerCaller {

  private final static Logger logger = LoggerFactory.getLogger(CalciteServerCaller.class);

  private static CalciteServerHandler handler;

  private static CalciteServer.Processor processor;

  private static int port=9093;

  public static void main(String[] args) {
    try {
      if (args.length == 1){
        port = Integer.valueOf(args[0]);
      }
      handler = new CalciteServerHandler();
      processor = new CalciteServer.Processor(handler);

      Runnable simple = new Runnable() {
        public void run() {
          simple(processor);
        }
      };

      new Thread(simple).start();
    } catch (Exception x) {
      x.printStackTrace();
    }
  }

  public static void simple(CalciteServer.Processor processor) {
    try {
      TServerTransport serverTransport = new TServerSocket(port);
      TServer server = new TSimpleServer(new Args(serverTransport).processor(processor));

      logger.info("Starting the simple server... Listening on port "+ port);
      handler.setServer(server);
      server.serve();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

}

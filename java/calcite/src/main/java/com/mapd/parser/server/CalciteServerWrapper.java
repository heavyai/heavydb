/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.parser.server;

import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TThreadPoolServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import com.mapd.thrift.calciteserver.CalciteServer.Processor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mapd.common.SockTransportProperties;
/**
 *
 * @author michael
 */
public class CalciteServerWrapper implements Runnable {
  private final static Logger MAPDLOGGER =
          LoggerFactory.getLogger(CalciteServerWrapper.class);
  private final CalciteServerHandler handler;
  private final Processor processor;
  private TServer server;
  private int mapDPort = 6274;
  private String dataDir = ("data/");
  private int calcitePort = 6279;
  private boolean shutdown = false;

  public CalciteServerWrapper() {
    handler = new CalciteServerHandler(mapDPort, dataDir, null, null);
    processor = new com.mapd.thrift.calciteserver.CalciteServer.Processor(handler);
  }

  public CalciteServerWrapper(int calcitePort,
          int mapDPort,
          String dataDir,
          String extensionFunctionsAstFile,
          SockTransportProperties skT) {
    handler = new CalciteServerHandler(mapDPort, dataDir, extensionFunctionsAstFile, skT);
    processor = new com.mapd.thrift.calciteserver.CalciteServer.Processor(handler);
    this.calcitePort = calcitePort;
    this.mapDPort = mapDPort;
  }

  private void startServer(
          com.mapd.thrift.calciteserver.CalciteServer.Processor processor) {
    try {
      TServerTransport serverTransport = new TServerSocket(calcitePort);
      server = new TThreadPoolServer(
              new TThreadPoolServer.Args(serverTransport).processor(processor));

      MAPDLOGGER.debug("Starting a threaded pool server... Listening on port "
              + calcitePort + " MapD on port " + mapDPort);
      handler.setServer(server);
      server.serve();
      // we have been told to shut down (only way to get to this piece of code
      shutdown = true;

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public void stopServer() {
    server.stop();
    shutdown = true;
  }

  @Override
  public void run() {
    startServer(processor);
  }

  boolean shutdown() {
    return shutdown;
  }
}

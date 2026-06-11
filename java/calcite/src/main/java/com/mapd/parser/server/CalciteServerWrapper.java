/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.parser.server;

import com.mapd.common.SockTransportProperties;

import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TThreadPoolServer;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TSSLTransportFactory.TSSLTransportParameters;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.heavy.thrift.calciteserver.CalciteServer.Processor;

public class CalciteServerWrapper implements Runnable {
  private final static Logger HEAVYDBLOGGER =
          LoggerFactory.getLogger(CalciteServerWrapper.class);
  private final CalciteServerHandler handler;
  private final Processor processor;
  private TServer server;
  private int heavyDBPort = 6274;
  private String dataDir = ("data/");
  private int calcitePort = 6279;
  private boolean shutdown = false;
  private SockTransportProperties server_skT_;

  public CalciteServerWrapper() {
    handler = new CalciteServerHandler(heavyDBPort, dataDir, null, null, "");
    processor = new ai.heavy.thrift.calciteserver.CalciteServer.Processor(handler);
  }

  public CalciteServerWrapper(int calcitePort,
          int heavyDBPort,
          String dataDir,
          String extensionFunctionsAstFile,
          SockTransportProperties client_skT,
          SockTransportProperties server_skT) {
    handler = new CalciteServerHandler(
            heavyDBPort, dataDir, extensionFunctionsAstFile, client_skT, "");
    processor = new ai.heavy.thrift.calciteserver.CalciteServer.Processor(handler);
    this.calcitePort = calcitePort;
    this.heavyDBPort = heavyDBPort;
    this.server_skT_ = server_skT;
  }

  public CalciteServerWrapper(int calcitePort,
          int heavyDBPort,
          String dataDir,
          String extensionFunctionsAstFile,
          SockTransportProperties client_skT,
          SockTransportProperties server_skT,
          String userDefinedFunctionsFile) {
    handler = new CalciteServerHandler(heavyDBPort,
            dataDir,
            extensionFunctionsAstFile,
            client_skT,
            userDefinedFunctionsFile);
    processor = new ai.heavy.thrift.calciteserver.CalciteServer.Processor(handler);
    this.calcitePort = calcitePort;
    this.heavyDBPort = heavyDBPort;
    this.server_skT_ = server_skT;
    try {
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private void startServer(
          ai.heavy.thrift.calciteserver.CalciteServer.Processor processor) {
    try {
      TServerTransport serverTransport = server_skT_.openServerTransport(calcitePort);
      server = new TThreadPoolServer(
              new TThreadPoolServer.Args(serverTransport).processor(processor));

      HEAVYDBLOGGER.debug("Starting a threaded pool server... Listening on port "
              + calcitePort + " HEAVY.AI on port " + heavyDBPort);
      handler.setServer(server);
      server.serve();
      // we have been told to shut down (only way to get to this piece of code
      shutdown = true;

    } catch (Exception e) {
      e.printStackTrace();
      HEAVYDBLOGGER.error(" Calcite server Failed to start ");
      shutdown = true;
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

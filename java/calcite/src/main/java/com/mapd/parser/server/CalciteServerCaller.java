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

import java.io.IOException;
import static java.lang.System.exit;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.PropertyConfigurator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.mapd.common.SockTransportProperties;

public class CalciteServerCaller {
  private SockTransportProperties skT = null;
  private final static Logger MAPDLOGGER =
          LoggerFactory.getLogger(CalciteServerCaller.class);
  private CommandLine cmd = null;

  public static void main(String[] args) {
    CalciteServerCaller csc = new CalciteServerCaller();
    csc.doWork(args);
  }

  private void doWork(String[] args) {
    CalciteServerWrapper calciteServerWrapper = null;

    // create Options object
    Options options = new Options();

    Option port =
            Option.builder("p").hasArg().desc("port number").longOpt("port").build();

    Option ssl_trust_store = Option.builder("T")
                                     .hasArg()
                                     .desc("SSL_trust_store")
                                     .longOpt("trust_store")
                                     .build();

    Option ssl_trust_passwd = Option.builder("P")
                                      .hasArg()
                                      .desc("SSL_trust_password")
                                      .longOpt("trust_store_pw")
                                      .build();

    Option mapdPort = Option.builder("m")
                              .hasArg()
                              .desc("mapd port number")
                              .longOpt("mapd_port")
                              .build();

    Option data = Option.builder("d")
                          .hasArg()
                          .desc("data directory")
                          .required()
                          .longOpt("data")
                          .build();

    Option extensions = Option.builder("e")
                                .hasArg()
                                .desc("extension signatures directory")
                                .longOpt("extensions")
                                .build();

    options.addOption(port);
    options.addOption(data);
    options.addOption(extensions);
    options.addOption(mapdPort);
    options.addOption(ssl_trust_store);
    options.addOption(ssl_trust_passwd);

    CommandLineParser parser = new DefaultParser();

    try {
      cmd = parser.parse(options, args);
    } catch (ParseException ex) {
      MAPDLOGGER.error(ex.getLocalizedMessage());
      help(options);
      exit(0);
    }

    int portNum = Integer.valueOf(cmd.getOptionValue("port", "6279"));
    int mapdPortNum = Integer.valueOf(cmd.getOptionValue("mapd_port", "6274"));
    String dataDir = cmd.getOptionValue("data", "data");
    String extensionsDir = cmd.getOptionValue("extensions", "build/QueryEngine");
    String trust_store = cmd.getOptionValue("trust_store", "");
    String trust_store_pw = cmd.getOptionValue("trust_store_pw", "");
    final Path extensionFunctionsAstFile =
            Paths.get(extensionsDir, "ExtensionFunctions.ast");
    // Add logging to our log files directories
    Properties p = new Properties();
    try {
      p.load(getClass().getResourceAsStream("/log4j.properties"));
    } catch (IOException ex) {
      MAPDLOGGER.error(
              "Could not load log4j property file from resources " + ex.getMessage());
    }
    p.put("log.dir", dataDir); // overwrite "log.dir"
    PropertyConfigurator.configure(p);

    try {
      if (!trust_store.isEmpty())
        skT = new SockTransportProperties(trust_store, trust_store_pw);
    } catch (Exception ex) {
      MAPDLOGGER.error(
              "Supplied java trust stored could not be opened " + ex.getMessage());
    }

    calciteServerWrapper = new CalciteServerWrapper(
            portNum, mapdPortNum, dataDir, extensionFunctionsAstFile.toString(), skT);

    while (true) {
      try {
        calciteServerWrapper.run();
        if (calciteServerWrapper.shutdown()) {
          break;
        }
        try {
          // wait for 4 secs before retry
          Thread.sleep(4000);
        } catch (InterruptedException ex) {
          // noop
        }
      } catch (Exception x) {
        x.printStackTrace();
      }
    }
  }

  private void help(Options options) {
    // automatically generate the help statement
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("CalciteServerCaller", options);
  }
}

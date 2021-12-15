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

import static java.lang.System.exit;

import com.mapd.common.SockTransportProperties;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
public class CalciteServerCaller {
  private SockTransportProperties client_skT = null;
  private SockTransportProperties server_skT = null;
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

    Option ssl_keystore =
            Option.builder("Y").hasArg().desc("SSL keystore").longOpt("keystore").build();

    Option ssl_keystore_password = Option.builder("Z")
                                           .hasArg()
                                           .desc("SSL keystore password")
                                           .longOpt("keystore_password")
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

    Option udf_file = Option.builder("u")
                              .hasArg()
                              .desc("User Defined Functions file path")
                              .longOpt("udf")
                              .build();

    Option config_file = Option.builder("c")
                                 .hasArg()
                                 .desc("Configuration file")
                                 .longOpt("config")
                                 .build();
    options.addOption(port);
    options.addOption(data);
    options.addOption(extensions);
    options.addOption(mapdPort);
    options.addOption(ssl_trust_store);
    options.addOption(ssl_trust_passwd);
    options.addOption(ssl_keystore);
    options.addOption(ssl_keystore_password);
    options.addOption(udf_file);
    options.addOption(config_file);

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
    String key_store = cmd.getOptionValue("keystore", "");
    String key_store_pw = cmd.getOptionValue("keystore_password", "");
    String udfName = cmd.getOptionValue("udf", "");
    String configuration_file = cmd.getOptionValue("config", "");

    final Path extensionFunctionsAstFile =
            Paths.get(extensionsDir, "ExtensionFunctions.ast");

    // Add logging to our log files directories
    Properties p = new Properties();
    try {
      p.load(getClass().getResourceAsStream("/log4j2.properties"));
    } catch (IOException ex) {
      MAPDLOGGER.error(
              "Could not load log4j property file from resources " + ex.getMessage());
    }
    p.put("log.dir", dataDir); // overwrite "log.dir"

    try {
      Properties properties = null;
      if (!configuration_file.isEmpty()) {
        properties = CalciteServerCaller.readPropertyFile(configuration_file);
      }
      if (trust_store == null || trust_store.isEmpty()) {
        client_skT = SockTransportProperties.getUnencryptedClient();
        // client_skT = new SockTransportProperties(load_trust_store);
      } else {
        if (properties != null && trust_store_pw.isEmpty()) {
          // If a configuration file was supplied and trust password is empty read
          // properties it to get the trust store password
          trust_store_pw = properties.getProperty("ssl-trust-password");
          if (trust_store_pw == null) {
            MAPDLOGGER.warn("Failed to load trust store password from config file ["
                    + configuration_file + "] for trust store [" + trust_store + "]");
          }
        }
        try {
          client_skT = SockTransportProperties.getEncryptedClientSpecifiedTrustStore(
                  trust_store, trust_store_pw);
        } catch (Exception eX) {
          String error =
                  "Loading encrypted client SockTransportProperties failed. Error - "
                  + eX.toString();
          throw new RuntimeException(error);
        }
      }
      if (key_store != null && !key_store.isEmpty()) {
        // If suppled a key store load a server transport, otherwise server_skT can stay
        // as null If a configuration file was supplied read and password is empty read
        // properties for key store password
        if (properties != null && key_store_pw.isEmpty()) {
          key_store_pw = properties.getProperty("ssl-keystore-password");
          if (key_store_pw == null) {
            String err = "Failed to load key store password from config file ["
                    + configuration_file + "] for key store [" + key_store + "]";
            throw new RuntimeException(err);
          }
        }
        try {
          server_skT =
                  SockTransportProperties.getEncryptedServer(key_store, key_store_pw);
        } catch (Exception eX) {
          String error =
                  "Loading encrypted Server SockTransportProperties failed. Error - "
                  + eX.toString();
          throw eX;
        }
      } else {
        server_skT = SockTransportProperties.getUnecryptedServer();
      }
    } catch (Exception ex) {
      MAPDLOGGER.error("Error opening SocketTransport. " + ex.getMessage());
      exit(0);
    }

    Path udfPath;

    try {
      if (!udfName.isEmpty()) {
        udfPath = Paths.get(udfName);
      }
    } catch (FileSystemNotFoundException ex1) {
      MAPDLOGGER.error("Could not load udf file " + ex1.getMessage());
    }

    calciteServerWrapper = new CalciteServerWrapper(portNum,
            mapdPortNum,
            dataDir,
            extensionFunctionsAstFile.toString(),
            client_skT,
            server_skT,
            udfName);

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

  static private Properties readPropertyFile(String fileName) {
    Properties properties = new Properties();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(fileName));
      StringBuffer sb = new StringBuffer();
      for (String line = bufferedReader.readLine(); line != null;
              line = bufferedReader.readLine()) {
        if (line.toLowerCase().equals("[web]")) {
          break;
        }
        line = line.replaceAll("[\",']", "");
        sb.append(line + "\n");
      }
      properties.load(new StringReader(sb.toString()));
    } catch (java.io.IOException iE) {
      MAPDLOGGER.warn("Could not load configuration file [" + fileName
              + "] to get keystore/truststore password. Error - " + iE.toString());
    }
    return properties;
  }
}

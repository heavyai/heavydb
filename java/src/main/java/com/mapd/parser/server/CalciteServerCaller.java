/*
 * Some cool MapD License
 */
package com.mapd.parser.server;

import com.mapd.utility.SQLImporter;
import static java.lang.System.exit;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CalciteServerCaller {

  private final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerCaller.class);
  private CommandLine cmd = null;

  public static void main(String[] args) {
    CalciteServerCaller csc = new CalciteServerCaller();
    csc.doWork(args);
  }

  private void doWork(String[] args) {
    CalciteServerWrapper calciteServerWrapper = null;

    // create Options object
    Options options = new Options();

    Option port = Option.builder("p")
            .hasArg()
            .desc("port number")
            .longOpt("port")
            .build();

    Option data = Option.builder("d")
            .hasArg()
            .desc("data directory")
            .required()
            .longOpt("data")
            .build();

    options.addOption(port);
    options.addOption(data);

    CommandLineParser parser = new DefaultParser();

    try {
      cmd = parser.parse(options, args);
    } catch (ParseException ex) {
      MAPDLOGGER.error(ex.getLocalizedMessage());
      help(options);
      exit(0);
    }

    int portNum = Integer.valueOf(cmd.getOptionValue("port", "9093"));
    String dataDir = cmd.getOptionValue("data", "data");

    calciteServerWrapper = new CalciteServerWrapper(portNum, -1, dataDir);

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

  private void help(Options options) {
    // automatically generate the help statement
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("SQLImporter", options);
  }

}
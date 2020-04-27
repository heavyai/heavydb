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
package com.mapd.utility;

import static java.lang.Math.pow;
import static java.lang.System.exit;

import com.mapd.common.SockTransportProperties;
import com.mapd.utility.db_vendors.Db_vendor_types;
import com.omnisci.thrift.server.*;

import org.apache.commons.cli.*;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.security.KeyStore;
import java.sql.*;
import java.time.*;
import java.util.ArrayList;
import java.util.List;

interface DateTimeUtils {
  long getSecondsFromMilliseconds(long milliseconds);
}

class MutuallyExlusiveOptionsException extends ParseException {
  protected MutuallyExlusiveOptionsException(String message) {
    super(message);
  }

  public static MutuallyExlusiveOptionsException create(String errMsg, String[] strings) {
    StringBuffer sb = new StringBuffer(
            "Mutually exclusive options used. " + errMsg + ". Options provided [");
    for (String s : strings) {
      sb.append(s);
      sb.append(" ");
    }
    sb.setCharAt(sb.length() - 1, ']');
    return new MutuallyExlusiveOptionsException(sb.toString());
  }
}
class SQLImporter_args {
  private Options options = new Options();

  void printVersion() {
    System.out.println("SQLImporter Version 4.6.0");
  }

  void printHelpMessage() {
    StringBuffer sb = new StringBuffer("\nSQLImporter ");
    // Ready for PKI auth
    // sb.append("(-u <userid> -p <password> | --client-cert <key store filename>
    sb.append("-u <userid> -p <password> [(--binary|--http|--https [--insecure])]\n");
    sb.append("-s <omnisci server host> -db <omnisci db> --port <omnisci server port>\n");
    // sb.append("([--ca-trust-store <ca trust store file name>]
    // --ca-trust-store-password
    // <trust store password> | --insecure)\n");
    sb.append(
            "[-d <other database JDBC drive class>] -c <other database JDBC connection string>\n");
    sb.append(
            "-su <other database user> -sp <other database user password> -su <other database sql statement>\n");
    sb.append(
            "-t <OmniSci target table> -b <transfer buffer size> -f <table fragment size>\n");
    sb.append("[-tr] -i <init commands file>\n");
    sb.append("\nSQLImporter -h | --help\n\n");

    HelpFormatter formatter = new HelpFormatter();
    // Forces help to print out options in order they were added rather
    // than in alphabetical order
    formatter.setOptionComparator(null);
    int help_width = 100;
    formatter.printHelp(help_width, sb.toString(), "", options, "");
  }

  SQLImporter_args() {
    options.addOption("r", true, "Row Load Limit");

    // OmniSci authentication options
    options.addOption(Option.builder("h").desc("help message").longOpt("help").build());
    options.addOption(
            Option.builder("u").hasArg().desc("OmniSci User").longOpt("user").build());
    options.addOption(Option.builder("p")
                              .hasArg()
                              .desc("OmniSci Password")
                              .longOpt("passwd")
                              .build());
    // OmniSci transport options
    OptionGroup transport_grp = new OptionGroup();
    transport_grp.addOption(Option.builder()
                                    .desc("use binary transport to connect to OmniSci ")
                                    .longOpt("binary")
                                    .build());
    transport_grp.addOption(Option.builder()
                                    .desc("use http transport to connect to OmniSci ")
                                    .longOpt("http")
                                    .build());
    transport_grp.addOption(Option.builder()
                                    .desc("use https transport to connect to OmniSci ")
                                    .longOpt("https")
                                    .build());
    options.addOptionGroup(transport_grp);

    // OmniSci database server details
    options.addOption(Option.builder("s")
                              .hasArg()
                              .desc("OmniSci Server")
                              .longOpt("server")
                              .build());
    options.addOption(Option.builder("db")
                              .hasArg()
                              .desc("OmniSci Database")
                              .longOpt("database")
                              .build());
    options.addOption(
            Option.builder().hasArg().desc("OmniSci Port").longOpt("port").build());

    // OmniSci server authentication options
    options.addOption(Option.builder()
                              .hasArg()
                              .desc("CA certificate trust store")
                              .longOpt("ca-trust-store")
                              .build());
    options.addOption(Option.builder()
                              .hasArg()
                              .desc("CA certificate trust store password")
                              .longOpt("ca-trust-store-passwd")
                              .build());
    options.addOption(
            Option.builder()
                    .desc("Inseure TLS - do not validate server OmniSci server credentials")
                    .longOpt("insecure")
                    .build());

    // Other database connection details
    options.addOption(Option.builder("d")
                              .hasArg()
                              .desc("JDBC driver class")
                              .longOpt("driver")
                              .build());
    options.addOption(Option.builder("c")
                              .hasArg()
                              .desc("JDBC Connection string")
                              .longOpt("jdbcConnect")
                              .required()
                              .build());
    options.addOption(Option.builder("su")
                              .hasArg()
                              .desc("Source User")
                              .longOpt("sourceUser")
                              .required()
                              .build());
    options.addOption(Option.builder("sp")
                              .hasArg()
                              .desc("Source Password")
                              .longOpt("sourcePasswd")
                              .required()
                              .build());
    options.addOption(Option.builder("ss")
                              .hasArg()
                              .desc("SQL Select statement")
                              .longOpt("sqlStmt")
                              .required()
                              .build());

    options.addOption(Option.builder("t")
                              .hasArg()
                              .desc("OmniSci Target Table")
                              .longOpt("targetTable")
                              .required()
                              .build());

    options.addOption(Option.builder("b")
                              .hasArg()
                              .desc("transfer buffer size")
                              .longOpt("bufferSize")
                              .build());
    options.addOption(Option.builder("f")
                              .hasArg()
                              .desc("table fragment size")
                              .longOpt("fragmentSize")
                              .build());

    options.addOption(Option.builder("tr")
                              .desc("Truncate table if it exists")
                              .longOpt("truncate")
                              .build());
    options.addOption(Option.builder("i")
                              .hasArg()
                              .desc("File containing init command for DB")
                              .longOpt("initializeFile")
                              .build());
  }

  private Option setOptionRequired(Option option) {
    option.setRequired(true);
    return option;
  }

  public CommandLine parse(String[] args) throws ParseException {
    CommandLineParser clp = new DefaultParser() {
      public CommandLine parse(Options options, String[] strings) throws ParseException {
        Options helpOptions = new Options();
        helpOptions.addOption(
                Option.builder("h").desc("help message").longOpt("help").build());
        try {
          CommandLine cmd = super.parse(helpOptions, strings);
        } catch (UnrecognizedOptionException uE) {
        }
        if (cmd.hasOption("help")) {
          printHelpMessage();
          exit(0);
        }
        if (cmd.hasOption("version")) {
          printVersion();
          exit(0);
        }
        cmd = super.parse(options, strings);
        if (!cmd.hasOption("user") && !cmd.hasOption("client-cert")) {
          throw new MissingArgumentException(
                  "Must supply either an OmniSci db user or a user certificate");
        }
        // if user supplied must have password and visa versa
        if (cmd.hasOption("user") || cmd.hasOption("passwd")) {
          options.addOption(setOptionRequired(options.getOption("user")));
          options.addOption(setOptionRequired(options.getOption("passwd")));
          super.parse(options, strings);
        }

        // FUTURE USE FOR USER Auth if user client-cert supplied must have client-key
        // and
        // visa versa
        if (false) {
          if (cmd.hasOption("client-cert") || cmd.hasOption("client-key")) {
            options.addOption(setOptionRequired(options.getOption("ca-trust-store")));
            options.addOption(
                    setOptionRequired(options.getOption("ca-trust-store-password")));
            super.parse(options, strings);
          }
          if (options.getOption("user").isRequired()
                  && options.getOption("client-key").isRequired()) {
            MutuallyExlusiveOptionsException meo =
                    MutuallyExlusiveOptionsException.create(
                            "user/password can not be use with client-cert/client-key",
                            strings);
            throw meo;
          }

          if (cmd.hasOption("http")
                  || cmd.hasOption("binary")
                          && (cmd.hasOption("client-cert")
                                  || cmd.hasOption("client-key"))) {
            MutuallyExlusiveOptionsException meo = MutuallyExlusiveOptionsException.create(
                    "http|binary can not be use with ca-cert|client-cert|client-key",
                    strings);
          }
        }

        if (cmd.hasOption("insecure") && !cmd.hasOption("https")) {
          MutuallyExlusiveOptionsException meo = MutuallyExlusiveOptionsException.create(
                  "insecure can only be use with https", strings);
          throw meo;
        }

        return cmd;
      }

      public CommandLine parse(Options options, String[] strings, boolean b)
              throws ParseException {
        return null;
      }
    };
    return clp.parse(options, args);
  }
}

public class SQLImporter {
  protected String session = null;
  protected OmniSci.Client client = null;
  private CommandLine cmd = null;
  final static Logger LOGGER = LoggerFactory.getLogger(SQLImporter.class);
  private DateTimeUtils dateTimeUtils = (milliseconds) -> {
    return milliseconds / 1000;
  };

  Db_vendor_types vendor_types = null;

  public static void main(String[] args) {
    SQLImporter sq = new SQLImporter();
    sq.doWork(args);
  }

  void doWork(String[] args) {
    // create Options object

    SQLImporter_args s_args = new SQLImporter_args();

    try {
      cmd = s_args.parse(args);
    } catch (ParseException ex) {
      LOGGER.error(ex.getLocalizedMessage());
      s_args.printHelpMessage();
      exit(0);
    }
    executeQuery();
  }

  void executeQuery() {
    Connection conn = null;
    Statement stmt = null;

    long totalTime = 0;

    try {
      // Open a connection
      LOGGER.info("Connecting to database url :" + cmd.getOptionValue("jdbcConnect"));
      conn = DriverManager.getConnection(cmd.getOptionValue("jdbcConnect"),
              cmd.getOptionValue("sourceUser"),
              cmd.getOptionValue("sourcePasswd"));
      vendor_types = Db_vendor_types.Db_vendor_factory(cmd.getOptionValue("jdbcConnect"));
      long startTime = System.currentTimeMillis();

      // run init file script on targe DB if present
      if (cmd.hasOption("initializeFile")) {
        run_init(conn);
      }

      // set autocommit off to allow postgress to not load all results
      try {
        conn.setAutoCommit(false);
      } catch (SQLException se) {
        LOGGER.warn(
                "SQLException when attempting to setAutoCommit to false, jdbc driver probably doesnt support it.  Error is "
                + se.toString());
      }

      // Execute a query
      stmt = conn.createStatement();

      int bufferSize = Integer.valueOf(cmd.getOptionValue("bufferSize", "10000"));
      // set the jdbc fetch buffer size to reduce the amount of records being moved to
      // java from postgress
      stmt.setFetchSize(bufferSize);
      long timer;

      ResultSet rs = stmt.executeQuery(cmd.getOptionValue("sqlStmt"));

      // check if table already exists and is compatible in OmniSci with the query
      // metadata
      ResultSetMetaData md = rs.getMetaData();
      checkMapDTable(conn, md);

      timer = System.currentTimeMillis();

      long resultCount = 0;
      int bufferCount = 0;
      long total = 0;

      List<TColumn> cols = new ArrayList(md.getColumnCount());
      for (int i = 1; i <= md.getColumnCount(); i++) {
        TColumn col = setupBinaryColumn(i, md, bufferSize);
        cols.add(col);
      }

      // read data from old DB
      while (rs.next()) {
        for (int i = 1; i <= md.getColumnCount(); i++) {
          setColValue(rs,
                  cols.get(i - 1),
                  md.getColumnType(i),
                  i,
                  md.getScale(i),
                  md.getColumnTypeName(i));
        }
        resultCount++;
        bufferCount++;
        if (bufferCount == bufferSize) {
          bufferCount = 0;
          // send the buffer to mapD
          client.load_table_binary_columnar(
                  session, cmd.getOptionValue("targetTable"), cols); // old
          // recreate columnar store for use
          for (int i = 1; i <= md.getColumnCount(); i++) {
            resetBinaryColumn(i, md, bufferSize, cols.get(i - 1));
          }

          if (resultCount % 100000 == 0) {
            LOGGER.info("Imported " + resultCount + " records");
          }
        }
      }
      if (bufferCount > 0) {
        // send the LAST buffer to mapD
        client.load_table_binary_columnar(
                session, cmd.getOptionValue("targetTable"), cols);
        bufferCount = 0;
      }
      LOGGER.info("result set count is " + resultCount + " read time is "
              + (System.currentTimeMillis() - timer) + "ms");

      // Clean-up environment
      rs.close();
      stmt.close();

      totalTime = System.currentTimeMillis() - startTime;
      conn.close();
    } catch (SQLException se) {
      LOGGER.error("SQLException - " + se.toString());
      se.printStackTrace();
    } catch (TOmniSciException ex) {
      LOGGER.error("TOmniSciException - " + ex.toString());
      ex.printStackTrace();
    } catch (TException ex) {
      LOGGER.error("TException failed - " + ex.toString());
      ex.printStackTrace();
    } finally {
      // finally block used to close resources
      try {
        if (stmt != null) {
          stmt.close();
        }
      } catch (SQLException se2) {
      } // nothing we can do
      try {
        if (conn != null) {
          conn.close();
        }
      } catch (SQLException se) {
        LOGGER.error("SQlException in close - " + se.toString());
        se.printStackTrace();
      } // end finally try
    } // end try
  }

  private void run_init(Connection conn) {
    // attempt to open file
    String line = "";
    try {
      BufferedReader reader =
              new BufferedReader(new FileReader(cmd.getOptionValue("initializeFile")));
      Statement stmt = conn.createStatement();
      while ((line = reader.readLine()) != null) {
        if (line.isEmpty()) {
          continue;
        }
        LOGGER.info("Running : " + line);
        stmt.execute(line);
      }
      stmt.close();
      reader.close();
    } catch (IOException e) {
      LOGGER.error("Exception occurred trying to read initialize file: "
              + cmd.getOptionValue("initFile"));
      exit(1);
    } catch (SQLException e) {
      LOGGER.error(
              "Exception occurred trying to execute initialize file entry : " + line);
      exit(1);
    }
  }

  private void help(Options options) {
    // automatically generate the help statement
    HelpFormatter formatter = new HelpFormatter();
    formatter.setOptionComparator(null); // get options in the order they are created
    formatter.printHelp("SQLImporter", options);
  }

  private void checkMapDTable(Connection otherdb_conn, ResultSetMetaData md)
          throws SQLException {
    createMapDConnection();
    String tName = cmd.getOptionValue("targetTable");

    if (tableExists(tName)) {
      // check if we want to truncate
      if (cmd.hasOption("truncate")) {
        executeMapDCommand("Drop table " + tName);
        createMapDTable(otherdb_conn, md);
      } else {
        List<TColumnType> columnInfo = getColumnInfo(tName);
        verifyColumnSignaturesMatch(otherdb_conn, columnInfo, md);
      }
    } else {
      createMapDTable(otherdb_conn, md);
    }
  }

  private void verifyColumnSignaturesMatch(Connection otherdb_conn,
          List<TColumnType> dstColumns,
          ResultSetMetaData srcColumns) throws SQLException {
    if (srcColumns.getColumnCount() != dstColumns.size()) {
      LOGGER.error("Table sizes do not match: Destination " + dstColumns.size()
              + " versus Source " + srcColumns.getColumnCount());
      exit(1);
    }
    for (int i = 1; i <= dstColumns.size(); ++i) {
      if (!dstColumns.get(i - 1).getCol_name().equalsIgnoreCase(
                  srcColumns.getColumnName(i))) {
        LOGGER.error(
                "Destination table does not have matching column in same order for column number"
                + i + " destination column name is " + dstColumns.get(i - 1).col_name
                + " versus Select " + srcColumns.getColumnName(i));
        exit(1);
      }
      TDatumType dstType = dstColumns.get(i - 1).getCol_type().getType();
      int dstPrecision = dstColumns.get(i - 1).getCol_type().getPrecision();
      int dstScale = dstColumns.get(i - 1).getCol_type().getScale();
      int srcType = srcColumns.getColumnType(i);
      int srcPrecision = srcColumns.getPrecision(i);
      int srcScale = srcColumns.getScale(i);

      boolean match = false;
      switch (srcType) {
        case java.sql.Types.TINYINT:
          match |= dstType == TDatumType.TINYINT;
          // NOTE: it's okay to import smaller type to a bigger one,
          // so we just fall through and try to match the next type.
          // But the order of case statements is important here!
        case java.sql.Types.SMALLINT:
          match |= dstType == TDatumType.SMALLINT;
        case java.sql.Types.INTEGER:
          match |= dstType == TDatumType.INT;
        case java.sql.Types.BIGINT:
          match |= dstType == TDatumType.BIGINT;
          break;
        case java.sql.Types.DECIMAL:
        case java.sql.Types.NUMERIC:
          match = dstType == TDatumType.DECIMAL && dstPrecision == srcPrecision
                  && dstScale == srcScale;
          break;
        case java.sql.Types.FLOAT:
        case java.sql.Types.REAL:
          match |= dstType == TDatumType.FLOAT;
          // Fall through and try double
        case java.sql.Types.DOUBLE:
          match |= dstType == TDatumType.DOUBLE;
          break;
        case java.sql.Types.TIME:
          match = dstType == TDatumType.TIME;
          break;
        case java.sql.Types.TIMESTAMP:
          match = dstType == TDatumType.TIMESTAMP;
          break;
        case java.sql.Types.DATE:
          match = dstType == TDatumType.DATE;
          break;
        case java.sql.Types.BOOLEAN:
        case java.sql.Types
                .BIT: // deal with postgres treating boolean as bit... this will bite me
          match = dstType == TDatumType.BOOL;
          break;
        case java.sql.Types.NVARCHAR:
        case java.sql.Types.VARCHAR:
        case java.sql.Types.NCHAR:
        case java.sql.Types.CHAR:
        case java.sql.Types.LONGVARCHAR:
        case java.sql.Types.LONGNVARCHAR:
          match = dstType == TDatumType.STR;
          break;
        case java.sql.Types.OTHER:
          // NOTE: I ignore subtypes (geography vs geopetry vs none) here just because it
          // makes no difference for OmniSciDB at the moment
          Db_vendor_types.GisType gisType =
                  vendor_types.find_gis_type(otherdb_conn, srcColumns, i);
          if (gisType.srid != dstScale) {
            match = false;
            break;
          }
          switch (dstType) {
            case POINT:
              match = gisType.type.equalsIgnoreCase("POINT");
              break;
            case LINESTRING:
              match = gisType.type.equalsIgnoreCase("LINESTRING");
              break;
            case POLYGON:
              match = gisType.type.equalsIgnoreCase("POLYGON");
              break;
            case MULTIPOLYGON:
              match = gisType.type.equalsIgnoreCase("MULTIPOLYGON");
              break;
            default:
              LOGGER.error("Column type " + JDBCType.valueOf(srcType).getName()
                      + " not Supported");
              exit(1);
          }
          break;
        default:
          LOGGER.error("Column type " + JDBCType.valueOf(srcType).getName()
                  + " not Supported");
          exit(1);
      }
      if (!match) {
        LOGGER.error("Source and destination types for column "
                + srcColumns.getColumnName(i)
                + " do not match. Please make sure that type, precision and scale are exactly the same");
        exit(1);
      }
    }
  }

  private void createMapDTable(Connection otherdb_conn, ResultSetMetaData metaData) {
    StringBuilder sb = new StringBuilder();
    sb.append("Create table ").append(cmd.getOptionValue("targetTable")).append("(");

    // Now iterate the metadata
    try {
      for (int i = 1; i <= metaData.getColumnCount(); i++) {
        if (i > 1) {
          sb.append(",");
        }
        LOGGER.debug("Column name is " + metaData.getColumnName(i));
        LOGGER.debug("Column type is " + metaData.getColumnTypeName(i));
        LOGGER.debug("Column type is " + metaData.getColumnType(i));

        sb.append(metaData.getColumnName(i)).append(" ");
        int col_type = metaData.getColumnType(i);
        if (col_type == java.sql.Types.OTHER) {
          Db_vendor_types.GisType type =
                  vendor_types.find_gis_type(otherdb_conn, metaData, i);
          sb.append(Db_vendor_types.gis_type_to_str(type));
        } else {
          sb.append(getColType(metaData.getColumnType(i),
                  metaData.getPrecision(i),
                  metaData.getScale(i)));
        }
      }
      sb.append(")");

      if (Integer.valueOf(cmd.getOptionValue("fragmentSize", "0")) > 0) {
        sb.append(" with (fragment_size = ");
        sb.append(cmd.getOptionValue("fragmentSize", "0"));
        sb.append(")");
      }

    } catch (SQLException ex) {
      LOGGER.error("Error processing the metadata - " + ex.toString());
      exit(1);
    }

    executeMapDCommand(sb.toString());
  }

  private void createMapDConnection() {
    TTransport transport = null;
    TProtocol protocol = new TBinaryProtocol(transport);
    int port = Integer.valueOf(cmd.getOptionValue("port", "6274"));
    String server = cmd.getOptionValue("server", "localhost");
    try {
      // Uses default certificate stores.
      boolean load_trust_store = cmd.hasOption("https");
      SockTransportProperties skT = null;
      if (cmd.hasOption("https")) {
        skT = SockTransportProperties.getEncryptedClientDefaultTrustStore(
                !cmd.hasOption("insecure"));
        transport = skT.openHttpsClientTransport(server, port);
        transport.open();
        protocol = new TJSONProtocol(transport);
      } else if (cmd.hasOption("http")) {
        skT = SockTransportProperties.getUnencryptedClient();
        transport = skT.openHttpClientTransport(server, port);
        protocol = new TJSONProtocol(transport);
      } else {
        skT = SockTransportProperties.getUnencryptedClient();
        transport = skT.openClientTransport(server, port);
        transport.open();
        protocol = new TBinaryProtocol(transport);
      }

      client = new OmniSci.Client(protocol);
      // This if will be useless until PKI signon
      if (cmd.hasOption("user")) {
        session = client.connect(cmd.getOptionValue("user", "admin"),
                cmd.getOptionValue("passwd", "HyperInteractive"),
                cmd.getOptionValue("database", "omnisci"));
      }
      LOGGER.debug("Connected session is " + session);

    } catch (TTransportException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(1);
    } catch (TOmniSciException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(2);
    } catch (TException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(3);
    } catch (Exception ex) {
      LOGGER.error("General exception - " + ex.toString());
      exit(4);
    }
  }

  private List<TColumnType> getColumnInfo(String tName) {
    LOGGER.debug("Getting columns for  " + tName);
    List<TColumnType> row_descriptor = null;
    try {
      TTableDetails table_details = client.get_table_details(session, tName);
      row_descriptor = table_details.row_desc;
    } catch (TOmniSciException ex) {
      LOGGER.error("column check failed - " + ex.toString());
      exit(3);
    } catch (TException ex) {
      LOGGER.error("column check failed - " + ex.toString());
      exit(3);
    }
    return row_descriptor;
  }

  private boolean tableExists(String tName) {
    LOGGER.debug("Check for table " + tName);
    try {
      List<String> recv_get_tables = client.get_tables(session);
      for (String s : recv_get_tables) {
        if (s.equals(tName)) {
          return true;
        }
      }
    } catch (TOmniSciException ex) {
      LOGGER.error("Table check failed - " + ex.toString());
      exit(3);
    } catch (TException ex) {
      LOGGER.error("Table check failed - " + ex.toString());
      exit(3);
    }
    return false;
  }

  private void executeMapDCommand(String sql) {
    LOGGER.info(" run comamnd :" + sql);

    try {
      TQueryResult sqlResult = client.sql_execute(session, sql + ";", true, null, -1, -1);
    } catch (TOmniSciException ex) {
      LOGGER.error("SQL Execute failed - " + ex.toString());
      exit(1);
    } catch (TException ex) {
      LOGGER.error("SQL Execute failed - " + ex.toString());
      exit(1);
    }
  }

  private String getColType(int cType, int precision, int scale) {
    // Note - if cType is OTHER a earlier call will have been made
    // to try and work out the db vendors specific type.
    if (precision > 19) {
      precision = 19;
    }
    if (scale > 19) {
      scale = 18;
    }
    switch (cType) {
      case java.sql.Types.TINYINT:
        return ("TINYINT");
      case java.sql.Types.SMALLINT:
        return ("SMALLINT");
      case java.sql.Types.INTEGER:
        return ("INTEGER");
      case java.sql.Types.BIGINT:
        return ("BIGINT");
      case java.sql.Types.FLOAT:
        return ("FLOAT");
      case java.sql.Types.DECIMAL:
        return ("DECIMAL(" + precision + "," + scale + ")");
      case java.sql.Types.DOUBLE:
        return ("DOUBLE");
      case java.sql.Types.REAL:
        return ("REAL");
      case java.sql.Types.NUMERIC:
        return ("NUMERIC(" + precision + "," + scale + ")");
      case java.sql.Types.TIME:
        return ("TIME");
      case java.sql.Types.TIMESTAMP:
        return ("TIMESTAMP");
      case java.sql.Types.DATE:
        return ("DATE");
      case java.sql.Types.BOOLEAN:
      case java.sql.Types
              .BIT: // deal with postgress treating boolean as bit... this will bite me
        return ("BOOLEAN");
      case java.sql.Types.NVARCHAR:
      case java.sql.Types.VARCHAR:
      case java.sql.Types.NCHAR:
      case java.sql.Types.CHAR:
      case java.sql.Types.LONGVARCHAR:
      case java.sql.Types.LONGNVARCHAR:
        return ("TEXT ENCODING DICT");
      default:
        throw new AssertionError("Column type " + cType + " not Supported");
    }
  }

  private TColumn setupBinaryColumn(int i, ResultSetMetaData md, int bufferSize)
          throws SQLException {
    TColumn col = new TColumn();

    col.nulls = new ArrayList<Boolean>(bufferSize);

    col.data = new TColumnData();

    switch (md.getColumnType(i)) {
      case java.sql.Types.TINYINT:
      case java.sql.Types.SMALLINT:
      case java.sql.Types.INTEGER:
      case java.sql.Types.BIGINT:
      case java.sql.Types.TIME:
      case java.sql.Types.TIMESTAMP:
      case java.sql.Types
              .BIT: // deal with postgress treating boolean as bit... this will bite me
      case java.sql.Types.BOOLEAN:
      case java.sql.Types.DATE:
      case java.sql.Types.DECIMAL:
      case java.sql.Types.NUMERIC:
        col.data.int_col = new ArrayList<Long>(bufferSize);
        break;

      case java.sql.Types.FLOAT:
      case java.sql.Types.DOUBLE:
      case java.sql.Types.REAL:
        col.data.real_col = new ArrayList<Double>(bufferSize);
        break;

      case java.sql.Types.NVARCHAR:
      case java.sql.Types.VARCHAR:
      case java.sql.Types.NCHAR:
      case java.sql.Types.CHAR:
      case java.sql.Types.LONGVARCHAR:
      case java.sql.Types.LONGNVARCHAR:
      case java.sql.Types.OTHER:
        col.data.str_col = new ArrayList<String>(bufferSize);
        break;

      default:
        throw new AssertionError("Column type " + md.getColumnType(i) + " not Supported");
    }
    return col;
  }

  private void setColValue(ResultSet rs,
          TColumn col,
          int columnType,
          int colNum,
          int scale,
          String colTypeName) throws SQLException {
    switch (columnType) {
      case java.sql.Types
              .BIT: // deal with postgress treating boolean as bit... this will bite me
      case java.sql.Types.BOOLEAN:
        Boolean b = rs.getBoolean(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.int_col.add(0L);
        } else {
          col.nulls.add(Boolean.FALSE);
          col.data.int_col.add(b ? 1L : 0L);
        }
        break;

      case java.sql.Types.DECIMAL:
      case java.sql.Types.NUMERIC:
        BigDecimal bd = rs.getBigDecimal(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.int_col.add(0L);
        } else {
          col.nulls.add(Boolean.FALSE);
          col.data.int_col.add(bd.multiply(new BigDecimal(pow(10L, scale))).longValue());
        }
        break;

      case java.sql.Types.TINYINT:
      case java.sql.Types.SMALLINT:
      case java.sql.Types.INTEGER:
      case java.sql.Types.BIGINT:
        Long l = rs.getLong(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.int_col.add(new Long(0));
        } else {
          col.nulls.add(Boolean.FALSE);
          col.data.int_col.add(l);
        }
        break;

      case java.sql.Types.TIME:
        Time t = rs.getTime(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.int_col.add(0L);

        } else {
          col.data.int_col.add(dateTimeUtils.getSecondsFromMilliseconds(t.getTime()));
          col.nulls.add(Boolean.FALSE);
        }

        break;
      case java.sql.Types.TIMESTAMP:
        Timestamp ts = rs.getTimestamp(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.int_col.add(0L);

        } else {
          col.data.int_col.add(dateTimeUtils.getSecondsFromMilliseconds(ts.getTime()));
          col.nulls.add(Boolean.FALSE);
        }

        break;
      case java.sql.Types.DATE:
        Date d = rs.getDate(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.int_col.add(0L);

        } else {
          col.data.int_col.add(dateTimeUtils.getSecondsFromMilliseconds(d.getTime()));
          col.nulls.add(Boolean.FALSE);
        }
        break;
      case java.sql.Types.FLOAT:
      case java.sql.Types.DOUBLE:
      case java.sql.Types.REAL:
        Double db = rs.getDouble(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.real_col.add(new Double(0));

        } else {
          col.nulls.add(Boolean.FALSE);
          col.data.real_col.add(db);
        }
        break;

      case java.sql.Types.NVARCHAR:
      case java.sql.Types.VARCHAR:
      case java.sql.Types.NCHAR:
      case java.sql.Types.CHAR:
      case java.sql.Types.LONGVARCHAR:
      case java.sql.Types.LONGNVARCHAR:
        String strVal = rs.getString(colNum);
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.str_col.add("");

        } else {
          col.data.str_col.add(strVal);
          col.nulls.add(Boolean.FALSE);
        }
        break;
      case java.sql.Types.OTHER:
        if (rs.wasNull()) {
          col.nulls.add(Boolean.TRUE);
          col.data.str_col.add("");
        } else {
          col.data.str_col.add(vendor_types.get_wkt(rs, colNum, colTypeName));
          col.nulls.add(Boolean.FALSE);
        }
        break;
      default:
        throw new AssertionError("Column type " + columnType + " not Supported");
    }
  }

  private void resetBinaryColumn(int i, ResultSetMetaData md, int bufferSize, TColumn col)
          throws SQLException {
    col.nulls.clear();

    switch (md.getColumnType(i)) {
      case java.sql.Types.TINYINT:
      case java.sql.Types.SMALLINT:
      case java.sql.Types.INTEGER:
      case java.sql.Types.BIGINT:
      case java.sql.Types.TIME:
      case java.sql.Types.TIMESTAMP:
      case java.sql.Types
              .BIT: // deal with postgress treating boolean as bit... this will bite me
      case java.sql.Types.BOOLEAN:
      case java.sql.Types.DATE:
      case java.sql.Types.DECIMAL:
      case java.sql.Types.NUMERIC:
        col.data.int_col.clear();
        break;

      case java.sql.Types.FLOAT:
      case java.sql.Types.DOUBLE:
      case java.sql.Types.REAL:
        col.data.real_col.clear();
        break;

      case java.sql.Types.NVARCHAR:
      case java.sql.Types.VARCHAR:
      case java.sql.Types.NCHAR:
      case java.sql.Types.CHAR:
      case java.sql.Types.LONGVARCHAR:
      case java.sql.Types.LONGNVARCHAR:
      case java.sql.Types.OTHER:
        col.data.str_col.clear();
        break;
      default:
        throw new AssertionError("Column type " + md.getColumnType(i) + " not Supported");
    }
  }
}

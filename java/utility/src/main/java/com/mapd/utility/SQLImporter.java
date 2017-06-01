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

import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TStringRow;
import com.mapd.thrift.server.TStringValue;
import com.mapd.thrift.server.TTableDetails;
import com.mapd.thrift.server.TMapDException;
import static java.lang.System.exit;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SQLImporter {

  protected String session = null;
  protected MapD.Client client = null;
  private CommandLine cmd = null;
  final static Logger LOGGER = LoggerFactory.getLogger(SQLImporter.class);

  public static void main(String[] args) {
    SQLImporter sq = new SQLImporter();
    sq.doWork(args);
  }

  void doWork(String[] args) {

    // create Options object
    Options options = new Options();

    // add r option
    options.addOption("r", true, "Row Load Limit");

    Option driver = Option.builder("d")
            .hasArg()
            .desc("JDBC driver class")
            .longOpt("driver")
            .build();

    Option sqlStmt = Option.builder("ss")
            .hasArg()
            .desc("SQL Select statement")
            .longOpt("sqlStmt")
            .required()
            .build();

    Option jdbcConnect = Option.builder("c")
            .hasArg()
            .desc("JDBC Connection string")
            .longOpt("jdbcConnect")
            .required()
            .build();

    Option user = Option.builder("u")
            .hasArg()
            .desc("MapD User")
            .longOpt("user")
            .build();

    Option sourceUser = Option.builder("su")
            .hasArg()
            .desc("Source User")
            .longOpt("sourceUser")
            .required()
            .build();

    Option sourcePasswd = Option.builder("sp")
            .hasArg()
            .desc("Source Password")
            .longOpt("sourcePasswd")
            .required()
            .build();

    Option passwd = Option.builder("p")
            .hasArg()
            .desc("MapD Password")
            .longOpt("passwd")
            .build();

    Option server = Option.builder("s")
            .hasArg()
            .desc("MapD Server")
            .longOpt("server")
            .build();

    Option targetTable = Option.builder("t")
            .hasArg()
            .desc("MapD Target Table")
            .longOpt("targetTable")
            .required()
            .build();

    Option port = Option.builder()
            .hasArg()
            .desc("MapD Port")
            .longOpt("port")
            .build();

    Option bufferSize = Option.builder("b")
            .hasArg()
            .desc("transfer buffer size")
            .longOpt("bufferSize")
            .build();

    Option fragmentSize = Option.builder("f")
            .hasArg()
            .desc("table fragment size")
            .longOpt("fragmentSize")
            .build();

    Option database = Option.builder("db")
            .hasArg()
            .desc("MapD Database")
            .longOpt("database")
            .build();

    Option truncate = Option.builder("tr")
            .desc("Truncate table if it exists")
            .longOpt("truncate")
            .build();

    options.addOption(driver);
    options.addOption(sqlStmt);
    options.addOption(jdbcConnect);
    options.addOption(user);
    options.addOption(server);
    options.addOption(passwd);
    options.addOption(port);
    options.addOption(sourceUser);
    options.addOption(sourcePasswd);
    options.addOption(targetTable);
    options.addOption(database);
    options.addOption(bufferSize);
    options.addOption(fragmentSize);
    options.addOption(truncate);

    CommandLineParser parser = new DefaultParser();

    try {
      cmd = parser.parse(options, args);
    } catch (ParseException ex) {
      LOGGER.error(ex.getLocalizedMessage());
      help(options);
      exit(0);
    }
    executeQuery();
  }

  void executeQuery() {
    Connection conn = null;
    Statement stmt = null;

    long totalTime = 0;

    try {
      //Open a connection
      LOGGER.info("Connecting to database url :" + cmd.getOptionValue("jdbcConnect"));
      conn = DriverManager.getConnection(cmd.getOptionValue("jdbcConnect"),
              cmd.getOptionValue("sourceUser"),
              cmd.getOptionValue("sourcePasswd"));

      long startTime = System.currentTimeMillis();

      // set autocommit off to allow postgress to not load all results
      conn.setAutoCommit(false);

      //Execute a query
      stmt = conn.createStatement();

      int bufferSize = Integer.valueOf(cmd.getOptionValue("bufferSize", "10000"));
      // set the jdbc fetch buffer size to reduce the amount of records being moved to java from postgress
      stmt.setFetchSize(bufferSize);
      long timer;

      ResultSet rs = stmt.executeQuery(cmd.getOptionValue("sqlStmt"));

      //check if table already exists and is compatible in MapD with the query metadata
      ResultSetMetaData md = rs.getMetaData();
      checkMapDTable(md);

      timer = System.currentTimeMillis();

      long resultCount = 0;
      int bufferCount = 0;
      long total = 0;

      List<TStringRow> rows = new ArrayList(bufferSize);
      while (rs.next()) {
        TStringRow tsr = new TStringRow();
        for (int i = 1; i <= md.getColumnCount(); i++) {
          // place string in rows array
          TStringValue tsv = new TStringValue();
          tsv.str_val = rs.getString(i);
          if (rs.wasNull()) {
            tsv.is_null = true;
          } else {
            tsv.is_null = false;
          }
          tsr.addToCols(tsv);
        }
        rows.add(tsr);

        resultCount++;
        bufferCount++;
        if (bufferCount == bufferSize) {
          bufferCount = 0;
          //send the buffer to mapD
          client.load_table(session, cmd.getOptionValue("targetTable"), rows);
          rows.clear();
          if (resultCount % 100000 == 0) {
            LOGGER.info("Imported " + resultCount + " records");
          }
        }
      }
      if (bufferCount > 0) {
        //send the LAST buffer to mapD
        client.load_table(session, cmd.getOptionValue("targetTable"), rows);
        rows.clear();
        bufferCount = 0;
      }
      LOGGER.info("result set count is " + resultCount + " read time is " + (System.currentTimeMillis() - timer) + "ms");

      //Clean-up environment
      rs.close();
      stmt.close();

      totalTime = System.currentTimeMillis() - startTime;
      conn.close();
    } catch (SQLException se) {
      LOGGER.error("SQLException - " + se.toString());
      se.printStackTrace();
    } catch (TMapDException ex) {
      LOGGER.error("TMapDException - " + ex.toString());
      ex.printStackTrace();
    } catch (TException ex) {
      LOGGER.error("TException failed - " + ex.toString());
      ex.printStackTrace();
    } finally {
      //finally block used to close resources
      try {
        if (stmt != null) {
          stmt.close();
        }
      } catch (SQLException se2) {
      }// nothing we can do
      try {
        if (conn != null) {
          conn.close();
        }
      } catch (SQLException se) {
        LOGGER.error("SQlException in close - " + se.toString());
        se.printStackTrace();
      }//end finally try
    }//end try
  }

  private void help(Options options) {
    // automatically generate the help statement
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("SQLImporter", options);
  }

  private void checkMapDTable(ResultSetMetaData md) throws SQLException {
    createMapDConnection();
    String tName = cmd.getOptionValue("targetTable");

    if (tableExists(tName)) {
      // check if we want to truncate
      if (cmd.hasOption("truncate")) {
        executeMapDCommand("Drop table " + tName);
        createMapDTable(md);
      } else {
        List<TColumnType> columnInfo = getColumnInfo(tName);
        // table exists lets check it has same number of columns
        if (md.getColumnCount() != columnInfo.size()) {
          LOGGER.error("Table sizes do not match - Mapd " + columnInfo.size() + " versus Select " + md.getColumnCount());
          exit(1);
        }
        // table exists lets check it is same layout - check names will do for now
        for (int colNum = 1; colNum <= columnInfo.size(); colNum++) {
          if (!columnInfo.get(colNum - 1).col_name.equalsIgnoreCase(md.getColumnName(colNum))) {
            LOGGER.error("MapD Table does not have matching column in same order for column number"
                    + colNum + " MapD column name is " + columnInfo.get(colNum - 1).col_name
                    + " versus Select " + md.getColumnName(colNum));
            exit(1);
          }
        }
      }
    } else {
      createMapDTable(md);
    }
  }

  private void createMapDTable(ResultSetMetaData metaData) {

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

        sb.append(getColType(metaData.getColumnType(i), metaData.getPrecision(i),
                metaData.getScale(i)));
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
    TTransport transport;
    try {
      transport = new TSocket(cmd.getOptionValue("server", "localhost"),
              Integer.valueOf(cmd.getOptionValue("port", "9091")));

      transport.open();

      TProtocol protocol = new TBinaryProtocol(transport);

      client = new MapD.Client(protocol);

      session = client.connect(cmd.getOptionValue("user", "mapd"),
              cmd.getOptionValue("passwd", "HyperInteractive"),
              cmd.getOptionValue("database", "mapd"));

      LOGGER.debug("Connected session is " + session);

    } catch (TTransportException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(1);
    } catch (TMapDException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(2);
    } catch (TException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(3);
    }
  }

  private List<TColumnType> getColumnInfo(String tName) {
    LOGGER.debug("Getting columns for  " + tName);
    List<TColumnType> row_descriptor = null;
    try {
      TTableDetails table_details = client.get_table_details(session, tName);
      row_descriptor = table_details.row_desc;
    } catch (TMapDException ex) {
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
    } catch (TMapDException ex) {
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
      TQueryResult sqlResult = client.sql_execute(session, sql + ";", true, null, -1);
    } catch (TMapDException ex) {
      LOGGER.error("SQL Execute failed - " + ex.toString());
      exit(1);
    } catch (TException ex) {
      LOGGER.error("SQL Execute failed - " + ex.toString());
      exit(1);
    }
  }

  private String getColType(int cType, int precision, int scale) {
    if (precision > 19) {
      precision = 19;
    }
    if (scale > 19) {
      scale = 18;
    }
    switch (cType) {
      case java.sql.Types.TINYINT:
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
      case java.sql.Types.BIT:  // deal with postgress treating boolean as bit... this will bite me
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

}

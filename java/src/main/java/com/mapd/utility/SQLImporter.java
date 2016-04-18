package com.mapd.utility;

/*
 * cool mapd License
 */
import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TStringRow;
import com.mapd.thrift.server.TStringValue;
import com.mapd.thrift.server.ThriftException;
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

  protected int session = 0;
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

    Option database = Option.builder("db")
            .hasArg()
            .desc("MapD Database")
            .longOpt("database")
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

      //Execute a query
      stmt = conn.createStatement();

      long timer = System.currentTimeMillis();
      ResultSet rs = stmt.executeQuery(cmd.getOptionValue("sqlStmt"));

      //produce new table in MapD with the query metadata
      ResultSetMetaData md = rs.getMetaData();
      createMapDTable(md);

      timer = System.currentTimeMillis();

      int resultCount = 0;
      int bufferCount = 0;
      int bufferSize = Integer.valueOf(cmd.getOptionValue("bufferSize", "10000"));

      List<TStringRow> rows = new ArrayList(bufferSize);
      while (rs.next()) {
        TStringRow tsr = new TStringRow();
        for (int i = 1; i <= md.getColumnCount(); i++){
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
        if (bufferCount == bufferSize){
          bufferCount = 0;
          //send the buffer to mapD
          client.load_table(session, cmd.getOptionValue("targetTable"), rows);
          rows.clear();
        }
      }
      if (bufferCount > 0){
          bufferCount = 0;
          //send the LAST buffer to mapD
          client.load_table(session, cmd.getOptionValue("targetTable"), rows);
          rows.clear();
        }
      LOGGER.info("result set count is " + resultCount + " read time is " + (System.currentTimeMillis() - timer)+ "ms");

      //Clean-up environment
      rs.close();
      stmt.close();

      totalTime = System.currentTimeMillis() - startTime;
      conn.close();
    } catch (SQLException se) {
      //Handle errors for JDBC
      se.printStackTrace();
    } catch (Exception e) {
      //Handle errors for Class.forName
      e.printStackTrace();
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
        se.printStackTrace();
      }//end finally try
    }//end try
  }

  private void help(Options options) {
    // automatically generate the help statement
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("SQLImporter", options);
  }

  private void createMapDTable(ResultSetMetaData metaData) {
    createMapDConnection();

    if (tableExists(cmd.getOptionValue("targetTable"))) {
      executeMapDCommand("Drop table " + cmd.getOptionValue("targetTable"));
    }

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

        sb.append(metaData.getColumnName(i)).append(" ").append(getColType(metaData.getColumnTypeName(i)));
      }
      sb.append(")");

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
    } catch (ThriftException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(2);
    } catch (TException ex) {
      LOGGER.error("Connection failed - " + ex.toString());
      exit(3);
    }
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
    } catch (ThriftException ex) {
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
      TQueryResult sqlResult = client.sql_execute(session, sql + ";", true, null);
    } catch (ThriftException ex) {
      LOGGER.error("SQL Execute failed - " + ex.toString());
      exit(1);
    } catch (TException ex) {
      LOGGER.error("SQL Execute failed - " + ex.toString());
      exit(1);
    }

  }

  private String getColType(String cType) {
    if (cType.equals("VARCHAR") || cType.equals("CHAR")) {
      return ("TEXT ENCODING DICT");
    } else {
      return cType;
    }
  }
}

/*
 *  Some cool MapD License
 */
package com.mapd.parser.server;

import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDUser;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.commons.pool.PoolableObjectFactory;
import org.apache.commons.pool.impl.GenericObjectPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class CalciteDirect {
final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteDirect.class);

  private final int mapDPort;

  private volatile long callCount;

  private final GenericObjectPool parserPool;

  public CalciteDirect(int port, String dataDir, String extensionFunctionsAstFile) {
    MAPDLOGGER.info("CalciteDirect Constructor port is '" + port + "' data dir is '" + dataDir +"'");
    this.parserPool = new GenericObjectPool();
    this.mapDPort = port;

    PoolableObjectFactory parserFactory = new CalciteParserFactory(dataDir, null);

    parserPool.setFactory(parserFactory);
    parserPool.setTestOnReturn(true);
  }

  public long getCallcount(){
    return callCount;
  }

  public void testNS(String[] args) {
    MAPDLOGGER.error("In Test NS:"+args.length);
    if (1 <= args.length) {
      MAPDLOGGER.error("Test call CalciteDirect from C++ : catalog dir is " + args[0]);
    }
  }

  public static void test(String[] args) {
    System.out.println("Hello, STATIC world!");
    if (1 <= args.length) {
      System.out.println(args[0]);
    }
  }

  public void updateMetadata(String jsonMetatData){
    MAPDLOGGER.info("Received new metadata from server");
  }

  public CalciteReturn process(String user, String passwd, String catalog, String sqlText, boolean legacySyntax) {
    MAPDLOGGER.debug(user + " " + " " + catalog +" '"+sqlText + "' " + legacySyntax);
    long timer = System.currentTimeMillis();
    callCount++;
    MapDParser parser;
    try {
      parser = (MapDParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool :" + ex.getMessage();
      MAPDLOGGER.error(msg);
      return new CalciteReturn("ERROR-- " +  msg, System.currentTimeMillis() - timer, true);
    }
    MapDUser mapDUser = new MapDUser(user, passwd, catalog, mapDPort);  //TODO MAT must fix so catalog can be scanned
    MAPDLOGGER.debug("process was called User:" + user + " Catalog:" + catalog + " sql :" + sqlText);

    // remove last charcter if it is a ;
    if (sqlText.charAt(sqlText.length() - 1) == ';') {
      sqlText = sqlText.substring(0, sqlText.length() - 1);
    }
    String relAlgebra;
    try {
      relAlgebra = parser.getRelAlgebra(sqlText, legacySyntax, mapDUser);
      MAPDLOGGER.debug("After get relalgebra");
    } catch (SqlParseException ex) {
      String msg = "Parse failed :" + ex.getPos() + ", " +  ex.getMessage();
      MAPDLOGGER.error(msg);
      return new CalciteReturn("ERROR-- " +  msg, System.currentTimeMillis() - timer, true);
    } catch (CalciteContextException ex) {
      String msg = "Validate failed :" + ex.getMessage();
      MAPDLOGGER.error(msg);
      return new CalciteReturn("ERROR-- " +  msg, System.currentTimeMillis() - timer, true);
    } catch (Exception ex) {
      String msg = "Exception Occured :" + ex.getMessage();
      ex.printStackTrace();
      MAPDLOGGER.error(msg);
      return new CalciteReturn("ERROR-- " +  msg, System.currentTimeMillis() - timer, true);
    } finally {
      try {
        // put parser object back in pool for others to use
        MAPDLOGGER.debug("Returning object to pool");
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object :" + ex.getMessage();
        MAPDLOGGER.error(msg);
        return new CalciteReturn("ERROR-- " +  msg, System.currentTimeMillis() - timer, true);
      }
    }
    MAPDLOGGER.debug("About to return good result");
    return new CalciteReturn(relAlgebra, System.currentTimeMillis() - timer, false);
  }
}

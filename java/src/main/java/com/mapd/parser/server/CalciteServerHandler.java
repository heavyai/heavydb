/*
 * Some cool MapD License
 */
package com.mapd.parser.server;

import com.mapd.calcite.parser.MapDParser;
import com.mapd.calcite.parser.MapDUser;
import com.mapd.thrift.calciteserver.InvalidParseRequest;
import com.mapd.thrift.calciteserver.TPlanResult;
import com.mapd.thrift.calciteserver.CalciteServer;
import java.io.IOException;
import java.util.Map;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.commons.pool.PoolableObjectFactory;
import org.apache.commons.pool.impl.GenericObjectPool;
import org.apache.thrift.TException;
import org.apache.thrift.server.TServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class CalciteServerHandler implements CalciteServer.Iface {

  final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteServerHandler.class);
  private TServer server;

  private final int mapDPort;

  private volatile long callCount;

  private final GenericObjectPool parserPool;

  private final String extSigsJson;

  CalciteServerHandler(int mapDPort, String dataDir, String extensionFunctionsAstFile) {
    this.parserPool = new GenericObjectPool();
    this.mapDPort = mapDPort;

    Map<String, ExtensionFunction> extSigs = null;
    try {
      extSigs = ExtensionFunctionSignatureParser.parse(extensionFunctionsAstFile);
    } catch (IOException ex) {
      MAPDLOGGER.error("Could not load extension function signatures: " + ex.getMessage());
    }
    this.extSigsJson = ExtensionFunctionSignatureParser.signaturesToJson(extSigs);

    PoolableObjectFactory parserFactory = new CalciteParserFactory(dataDir, null);

    parserPool.setFactory(parserFactory);
  }

  @Override
  public void ping() throws TException {
    MAPDLOGGER.info("Ping hit");
  }

  @Override
  public TPlanResult process(String user, String passwd, String catalog, String sqlText, boolean legacySyntax) throws InvalidParseRequest, TException {
    long timer = System.currentTimeMillis();
    callCount++;
    MapDParser parser;
    try {
      parser = (MapDParser) parserPool.borrowObject();
    } catch (Exception ex) {
      String msg = "Could not get Parse Item from pool :" + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-1, msg);
    }
    MapDUser mapDUser = new MapDUser(user, passwd, catalog, mapDPort);
    MAPDLOGGER.debug("process was called User:" + user + " Catalog:" + catalog + " sql :" + sqlText);

    // remove last charcter if it is a ;
    if (sqlText.charAt(sqlText.length() - 1) == ';') {
      sqlText = sqlText.substring(0, sqlText.length() - 1);
    }
    String relAlgebra;
    try {
      relAlgebra = parser.getRelAlgebra(sqlText, legacySyntax, mapDUser);
    } catch (SqlParseException ex) {
      String msg = "Parse failed :" + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-2, msg);
    } catch (CalciteContextException ex) {
      String msg = "Validate failed :" + ex.getMessage();
      MAPDLOGGER.error(msg);
      throw new InvalidParseRequest(-3, msg);
    } finally {
      try {
        // put parser object back in pool for others to use
        parserPool.returnObject(parser);
      } catch (Exception ex) {
        String msg = "Could not return parse object :" + ex.getMessage();
        MAPDLOGGER.error(msg);
        throw new InvalidParseRequest(-4, msg);
      }
    }
    return new TPlanResult(relAlgebra, System.currentTimeMillis() - timer);
  }

  @Override
  public void shutdown() throws TException {
    // received request to shutdown
    MAPDLOGGER.info("Shutdown calcite java server");
    server.stop();
  }

  @Override
  public String getExtensionFunctionWhitelist() {
    return this.extSigsJson;
  }

  void setServer(TServer s) {
    server = s;
  }
}

/*
 *  Some cool MapD License
 */
package com.mapd.parser.server;

import java.util.Map;

import com.mapd.calcite.parser.MapDParser;
import org.apache.commons.pool.PoolableObjectFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class CalciteParserFactory implements PoolableObjectFactory {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(CalciteParserFactory.class);

  private final String dataDir;
  private final Map<String, ExtensionFunction> extSigs;

  public CalciteParserFactory(String dataDir, final Map<String, ExtensionFunction> extSigs) {
    this.dataDir = dataDir;
    this.extSigs = extSigs;
  }

  @Override
  public Object makeObject() throws Exception {
    MapDParser obj = new MapDParser(dataDir, extSigs);
    return obj;
  }

  @Override
  public void destroyObject(Object obj) throws Exception {
    //no need to do anything
  }

  @Override
  public boolean validateObject(Object obj) {
    MapDParser mdp = (MapDParser)obj;
    if (mdp.getCallCount() < 1000) {
      return true;
    } else {
      MAPDLOGGER.debug(" invalidating object due to max use count");
      return false;
    }
  }

  @Override
  public void activateObject(Object obj) throws Exception {
    // don't need to do anything
  }

  @Override
  public void passivateObject(Object obj) throws Exception {
    // nothing to currently do here
  }

}

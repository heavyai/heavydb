/*
 *  Some cool MapD License
 */
package com.mapd.parser.server;

import com.mapd.calcite.parser.MapDParser;
import org.apache.commons.pool.PoolableObjectFactory;

/**
 *
 * @author michael
 */
class CalciteParserFactory implements PoolableObjectFactory {

  public CalciteParserFactory() {
  }

  @Override
  public Object makeObject() throws Exception {
    MapDParser obj = new MapDParser();
    return obj;
  }

  @Override
  public void destroyObject(Object obj) throws Exception {
    //no need to do anything
  }

  @Override
  public boolean validateObject(Object obj) {
    return true;
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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.parser.server;

import com.mapd.calcite.parser.CalciteParser;
import com.mapd.thrift.calciteserver.InvalidParseRequest;
import com.mapd.thrift.calciteserver.CalciteServer;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class CalciteServerHandler implements CalciteServer.Iface {

  final static Logger logger = LoggerFactory.getLogger(CalciteServerHandler.class);

  CalciteParser parser = new CalciteParser();

  @Override
  public void ping() throws TException {
    logger.info("Ping hit");
  }

  @Override
  public String process(String user, String passwd, String catalog, String sqlText) throws InvalidParseRequest, TException {
    logger.info("process was called User:"+user + " Catalog:"+ catalog+ "sql :" + sqlText);

    // remove last charcter if it is a ;
    if (sqlText.charAt(sqlText.length()-1) == ';'){
      sqlText = sqlText.substring(0, sqlText.length()-1);
    }
    String relAlgebra;
    try {
      relAlgebra = parser.getRelAlgebra(sqlText);
    } catch (SqlParseException ex) {
      logger.error("Parse failed :"+ ex.getMessage());
      throw new InvalidParseRequest(-1, ex.getMessage());
    } catch (CalciteContextException ex) {
      logger.error("Validate failed :"+ ex.getMessage());
      throw new InvalidParseRequest(-1, ex.getMessage());
    }

    return relAlgebra;
  }

}

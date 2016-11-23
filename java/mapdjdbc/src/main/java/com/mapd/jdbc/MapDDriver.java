/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.jdbc;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.DriverPropertyInfo;
import java.sql.SQLException;
import java.sql.SQLFeatureNotSupportedException;
import java.util.Properties;
import java.util.logging.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
public class MapDDriver implements java.sql.Driver {

  final static org.slf4j.Logger logger = LoggerFactory.getLogger(MapDDriver.class);
  public static final String PREFIX = "jdbc:mapd:";

  static {
    try {
      DriverManager.registerDriver(new MapDDriver());
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  @Override
  public Connection connect(String url, Properties info) throws SQLException { //logger.debug("Entered");
    if (!isValidURL(url)) {
      return null;
    }

    url = url.trim();
    return new MapDConnection(url, info);
  }

  @Override
  public boolean acceptsURL(String url) throws SQLException { //logger.debug("Entered");
    return isValidURL(url);
  }

  /**
   * Validates a URL
   *
   * @param url
   * @return true if the URL is valid, false otherwise
   */
  private static boolean isValidURL(String url) {
    return url != null && url.toLowerCase().startsWith(PREFIX);
  }

  @Override
  public DriverPropertyInfo[] getPropertyInfo(String url, Properties info) throws SQLException { //logger.debug("Entered");
    return null;
  }

  @Override
  public int getMajorVersion() {
    return 0;
  }

  @Override
  public int getMinorVersion() {
    return 1;

  }

  @Override
  public boolean jdbcCompliant() {
    return false;
  }

  @Override
  public Logger getParentLogger() throws SQLFeatureNotSupportedException {
    return null;
  }

}

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

package com.omnisci.jdbc;

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
public class OmniSciDriver implements java.sql.Driver {
  static final int DriverMajorVersion = 2;
  static final int DriverMinorVersion = 1;

  final static org.slf4j.Logger logger = LoggerFactory.getLogger(OmniSciDriver.class);
  public static final String OMNISCI_PREFIX = "jdbc:omnisci:";
  public static final String MAPD_PREFIX = "jdbc:mapd:";

  static {
    try {
      DriverManager.registerDriver(new OmniSciDriver());
    } catch (SQLException e) {
      e.printStackTrace();
    }
  }

  @Override
  public Connection connect(String url, Properties info)
          throws SQLException { // logger.debug("Entered");
    if (!isValidURL(url)) {
      return null;
    }

    url = url.trim();
    return new OmniSciConnection(url, info);
  }

  @Override
  public boolean acceptsURL(String url) throws SQLException { // logger.debug("Entered");
    return isValidURL(url);
  }

  /**
   * Validates a URL
   *
   * @param url
   * @return true if the URL is valid, false otherwise
   */
  private static boolean isValidURL(String url) {
    return url != null
            && (url.toLowerCase().startsWith(OMNISCI_PREFIX)
                       || url.toLowerCase().startsWith(MAPD_PREFIX));
  }

  @Override
  public DriverPropertyInfo[] getPropertyInfo(String url, Properties info)
          throws SQLException { // logger.debug("Entered");
    return null;
  }

  @Override
  public int getMajorVersion() {
    return DriverMajorVersion;
  }

  @Override
  public int getMinorVersion() {
    return DriverMinorVersion;
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

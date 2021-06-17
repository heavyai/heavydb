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

import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.DriverPropertyInfo;
import java.sql.SQLException;
import java.util.Properties;
import java.util.logging.Logger;

/**
 *
 * @author michael
 */
public class OmniSciDriver implements java.sql.Driver {
  static int DriverMajorVersion = -1;
  static int DriverMinorVersion = -1;
  static String DriverVersion = "UNKNOWN";
  final static String VERSION_FILE = "version.properties";
  final static org.slf4j.Logger logger = LoggerFactory.getLogger(OmniSciDriver.class);
  public static final String OMNISCI_PREFIX = "jdbc:omnisci:";
  public static final String MAPD_PREFIX = "jdbc:mapd:";

  static {
    try {
      DriverManager.registerDriver(new OmniSciDriver());
    } catch (SQLException e) {
      e.printStackTrace();
    }

    try (InputStream input = OmniSciDriver.class.getClassLoader().getResourceAsStream(
                 VERSION_FILE)) {
      Properties prop = new Properties();
      if (input == null) {
        logger.error("Cannot read " + VERSION_FILE + " file");
      } else {
        prop.load(input);
        DriverVersion = prop.getProperty("version");
        String[] version = DriverVersion.split("\\.");
        try {
          DriverMajorVersion = Integer.parseInt(version[0]);
          DriverMinorVersion = Integer.parseInt(version[1]);
        } catch (NumberFormatException ex) {
          logger.error("Unexpected driver version format in " + VERSION_FILE);
          DriverVersion = "UNKNOWN";
        }
      }
    } catch (IOException e) {
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
  public Logger getParentLogger() {
    return null;
  }
}

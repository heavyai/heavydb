package com.mapd.jdbc;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Property_loader extends Properties {
  public Property_loader() {
    try {
      ClassLoader cl = MapDDatabaseMetaData.class.getClassLoader();
      InputStream is = cl.getResourceAsStream("connection.properties");
      super.load(is);
    } catch (IOException io) {
      throw new RuntimeException(io.toString());
    }
  }
}

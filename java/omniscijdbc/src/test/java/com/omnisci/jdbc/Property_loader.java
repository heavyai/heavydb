package com.omnisci.jdbc;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Properties;

public class Property_loader extends Properties {
  public Property_loader(String property_file_name) {
    try {
      ClassLoader cl = getClass().getClassLoader();
      InputStream is = cl.getResourceAsStream(property_file_name);
      super.load(is);
    } catch (IOException io) {
      throw new RuntimeException(io.toString());
    }
  }
}

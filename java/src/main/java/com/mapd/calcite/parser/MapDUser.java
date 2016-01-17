/*
 *  Some cool MapD License
 */
package com.mapd.calcite.parser;

/**
 *
 * @author michael
 */
public class MapDUser {
  private final String user;
  private final String passwd;
  private final String catalog;
  private final int mapDPort;

  public MapDUser(String user, String passwd, String catalog, int mapDPort) {
    this.user = user;
    this.passwd = passwd;
    this.catalog = catalog;
    this.mapDPort = mapDPort;
  }

  public String getDB() {
    return catalog;
  }

  public String getPasswd() {
    return passwd;
  }

  public String getUser() {
    return user;
  }

  public int getMapDPort() {
    return mapDPort;
  }
}

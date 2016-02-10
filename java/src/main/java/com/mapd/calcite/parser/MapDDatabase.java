/*
 * Some cool MapD Header
 */
package com.mapd.calcite.parser;

import com.google.common.collect.Lists;
import static com.mapd.calcite.parser.MapDCatalogReader.DEFAULT_CATALOG;
import java.util.List;

/**
 *
 * @author michael
 */
/**
 * MapD schema.
 */
public class MapDDatabase {

  private final List<String> tableNames = Lists.newArrayList();
  private final String name;

  public MapDDatabase(String name) {
    this.name = name;
  }

  public void addTable(String name) {
    tableNames.add(name);
  }

  public String getCatalogName() {
    return DEFAULT_CATALOG;
  }

  String getSchemaName() {
    return name;
  }

  Iterable<String> getTableNames() {
    return tableNames;
  }
}

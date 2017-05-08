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

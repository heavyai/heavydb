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

import org.apache.calcite.rel.rules.Restriction;

/**
 *
 * @author michael
 */
public class MapDUser {
  private final String user;
  private final String catalog;
  private final Restriction restriction;

  public MapDUser(String user, String catalog, Restriction restriction) {
    this.user = user;
    this.catalog = catalog;
    this.restriction = restriction;
  }

  public Restriction getRestriction() {
    return restriction;
  }

  public String getDB() {
    return catalog;
  }

  public String getUser() {
    return user;
  }
}

/*
 * Copyright 2022 HEAVY.AI, Inc.
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

import java.util.List;

public class HeavyDBUser {
  private final String user;
  private final String catalog;
  private final int port;
  private final String session;
  private final List<Restriction> restrictions;

  public HeavyDBUser(String user,
          String session,
          String catalog,
          int port,
          List<Restriction> restrictions) {
    this.user = user;
    this.catalog = catalog;
    this.port = port;
    this.session = session;
    this.restrictions = restrictions;
  }

  public List<Restriction> getRestrictions() {
    return restrictions;
  }

  public String getDB() {
    return catalog;
  }

  public String getUser() {
    return user;
  }

  public int getPort() {
    return port;
  }

  public String getSession() {
    return session;
  }
}

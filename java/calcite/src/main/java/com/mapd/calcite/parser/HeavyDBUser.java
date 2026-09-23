/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

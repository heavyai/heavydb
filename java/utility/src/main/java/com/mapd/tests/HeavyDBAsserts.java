/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.mapd.tests;

import ai.heavy.thrift.server.TDashboard;

public class HeavyDBAsserts {
  public static interface TestRun { void run() throws Exception; }

  public static void assertEqual(Object a, Object b) {
    if (a.equals(b)) return;
    throw new RuntimeException("assert failed:\nExpected: " + a + "\n     got: " + b);
  }

  public static void assertEqual(int a, int b) {
    if (a == b) return;
    throw new RuntimeException("assert failed:\nExpected: " + a + "\n     got: " + b);
  }

  public static void assertEqual(String name, TDashboard db) {
    assertEqual(name, db.getDashboard_name());
    assertEqual(name + "_hash", db.getImage_hash());
    assertEqual(name + "_meta", db.getDashboard_metadata());
  }

  public static void shouldThrowException(String msg, TestRun test) {
    boolean failed;
    try {
      test.run();
      failed = true;
    } catch (Exception e) {
      failed = false;
    }

    if (failed) {
      throw new RuntimeException(msg);
    }
  }
}

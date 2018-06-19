/*
 * Copyright 2015 The Apache Software Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mapd.tests;

import com.mapd.thrift.server.TDashboard;

public class MapdAsserts {

  public static interface TestRun {
    void run() throws Exception;
  }

  public static void assertEqual(Object a, Object b) {
    if (a.equals(b))
      return;
    throw new RuntimeException("assert failed:\nExpected: "+a+"\n     got: " +b);
  }

  public static void assertEqual(int a, int b) {
    if (a == b)
      return;
    throw new RuntimeException("assert failed:\nExpected: "+a+"\n     got: " +b);
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

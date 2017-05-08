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

package com.mapd.parser.server;

/**
 *
 * @author michael
 */
public class CalciteReturn {
  private final String returnText;
  private final long elapsedTime;
  private final boolean failed;

  CalciteReturn(String string, long l, boolean b) {
    returnText = string;
    elapsedTime= l;
    failed = b;
  }

  public String getText() {
    return returnText;
  }

  public long getElapsedTime(){
    return elapsedTime;
  }

  public boolean hasFailed(){
    return failed;
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.heavy.jdbc;

public class HeavyAIExceptionText {
  static String getExceptionDetail(Exception ex) {
    if (ex.getStackTrace().length < 1) {
      return "Error in stack trace processing";
    }
    StackTraceElement sE = ex.getStackTrace()[0];
    return "[" + sE.getFileName() + ":" + sE.getMethodName() + ":" + sE.getLineNumber()
            + ":" + ex.toString() + "]";
  }
}

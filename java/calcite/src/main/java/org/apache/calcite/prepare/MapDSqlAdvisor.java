/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.calcite.prepare;

import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.sql.advise.SqlAdvisor;
import org.apache.calcite.sql.validate.SqlMoniker;
import org.apache.calcite.sql.validate.SqlMonikerImpl;
import org.apache.calcite.sql.validate.SqlMonikerType;

class MapDSqlAdvisor extends SqlAdvisor {
  public MapDSqlAdvisor(MapDSqlAdvisorValidator validator) {
    super(validator);
    this.permissionsAwareValidator = validator;
  }

  @Override
  public List<SqlMoniker> getCompletionHints(String sql, int cursor, String[] replaced) {
    // search backward starting from current position to find a "word"
    int wordStart = cursor;
    boolean quoted = false;
    while (wordStart > 0 && Character.isJavaIdentifierPart(sql.charAt(wordStart - 1))) {
      --wordStart;
    }
    if ((wordStart > 0) && (sql.charAt(wordStart - 1) == '"')) {
      quoted = true;
      --wordStart;
    }

    if (wordStart < 0) {
      return java.util.Collections.emptyList();
    }

    // Search forwards to the end of the word we should remove. Eat up
    // trailing double-quote, if any
    int wordEnd = cursor;
    while (wordEnd < sql.length() && Character.isJavaIdentifierPart(sql.charAt(wordEnd))) {
      ++wordEnd;
    }
    if (quoted && (wordEnd < sql.length()) && (sql.charAt(wordEnd) == '"')) {
      ++wordEnd;
    }

    // remove the partially composed identifier from the
    // sql statement - otherwise we get a parser exception
    String word = replaced[0] = sql.substring(wordStart, cursor);
    if (wordStart < wordEnd) {
      sql = sql.substring(0, wordStart) + sql.substring(wordEnd, sql.length());
    }

    // The table hints come from validator with a database prefix,
    // which is inconsistent with how tables are used in the query.
    List<SqlMoniker> completionHints = stripDatabaseFromTableHints(getCompletionHints0(sql, wordStart));

    if (permissionsAwareValidator.hasViolatedTablePermissions()) {
      return new ArrayList<>();
    }

    completionHints = applyPermissionsToTableHints(completionHints);

    // If cursor was part of the way through a word, only include hints
    // which start with that word in the result.
    final List<SqlMoniker> result;
    if (word.length() > 0) {
      result = new java.util.ArrayList<SqlMoniker>();
      if (quoted) {
        // Quoted identifier. Case-sensitive match.
        word = word.substring(1);
        for (SqlMoniker hint : completionHints) {
          String cname = hint.toString();
          if (cname.startsWith(word)) {
            result.add(hint);
          }
        }
      } else {
        // Regular identifier. Case-insensitive match.
        for (SqlMoniker hint : completionHints) {
          String cname = hint.toString();
          if ((cname.length() >= word.length()) && cname.substring(0, word.length()).equalsIgnoreCase(word)) {
            result.add(hint);
          }
        }
      }
    } else {
      result = completionHints;
    }

    return result;
  }

  private static List<SqlMoniker> stripDatabaseFromTableHints(final List<SqlMoniker> completionHints) {
    List<SqlMoniker> strippedCompletionHints = new ArrayList<>();
    for (final SqlMoniker hint : completionHints) {
      if (hint.getType() == SqlMonikerType.TABLE && hint.getFullyQualifiedNames().size() == 2) {
        final String tableName = hint.getFullyQualifiedNames().get(1);
        strippedCompletionHints.add(new SqlMonikerImpl(tableName, SqlMonikerType.TABLE));
      } else {
        strippedCompletionHints.add(hint);
      }
    }
    return strippedCompletionHints;
  }

  private List<SqlMoniker> applyPermissionsToTableHints(final List<SqlMoniker> completionHints) {
    List<SqlMoniker> completionHintsWithPermissions = new ArrayList<>();
    for (final SqlMoniker hint : completionHints) {
      if (hint.getType() == SqlMonikerType.TABLE) {
        // Database was stripped in previous step.
        assert hint.getFullyQualifiedNames().size() == 1;
        // Don't return tables which aren't visible per the permissions.
        if (permissionsAwareValidator.tableViolatesPermissions(hint.toString())) {
          continue;
        } else {
          completionHintsWithPermissions.add(hint);
        }
      } else {
        completionHintsWithPermissions.add(hint);
      }
    }
    return completionHintsWithPermissions;
  }

  private MapDSqlAdvisorValidator permissionsAwareValidator;
}

package com.omnisci.jdbc;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class OmniSciEscapeParser {
  private static final char[] QUOTE_OR_ALPHABETIC_MARKER = {'\"', '0'};
  // private static final char[] QUOTE_OR_ALPHABETIC_MARKER_OR_PARENTHESIS = {'\"', '0',
  // '('};
  private static final char[] SINGLE_QUOTE = {'\''};

  private enum EscapeFunctions {
    ESC_FUNCTION("\\s*fn\\s+([^ ]*?)\\s*\\(", "\\(\\s*(.*)\\s*\\)", null),
    ESC_DATE("\\s*(d)\\s+", "('.*?')", "DATE "),
    ESC_TIME("\\s*(t)\\s+", "('.*?')", "TIME "),
    ESC_TIMESTAMP("\\s*(ts)\\s+", "('.*?')", "TIMESTAMP ");

    EscapeFunctions(String escapePattern, String argPattern, String replacementKeyword) {
      this.escapePattern = Pattern.compile(escapePattern);
      this.argPattern = Pattern.compile(argPattern);
      this.replacementKeyword = replacementKeyword;
    }

    private final Pattern escapePattern;
    private final Pattern argPattern;
    private final String replacementKeyword;

    private String makeMatch(String sql) {
      Matcher matcher = escapePattern.matcher(sql);
      if (matcher.find()) {
        if (this == EscapeFunctions.ESC_DATE || this == EscapeFunctions.ESC_TIME
                || this == EscapeFunctions.ESC_TIMESTAMP) {
          matcher = argPattern.matcher(sql);
          if (matcher.find()) {
            String new_sql = this.replacementKeyword + matcher.group(1);
            return new_sql;
          }
        } else if (this == EscapeFunctions.ESC_FUNCTION) {
          String fn_name = matcher.group(1);
          Method method = OmniSciEscapeFunctions.getFunction(fn_name);
          matcher = argPattern.matcher(sql);
          if (matcher.find()) {
            if (method == null) {
              String new_sql = fn_name + '(' + matcher.group(1) + ')';
              return new_sql;
            } else {
              try {
                StringBuilder sb = new StringBuilder();
                List<CharSequence> parseArgs = new ArrayList<CharSequence>(3);
                String[] args = matcher.group(1).split(",");
                for (String s : args) {
                  parseArgs.add(s);
                }
                method.invoke(null, sb, parseArgs);
                return sb.toString();
              } catch (InvocationTargetException e) {
              } catch (IllegalAccessException ilE) {
              }
            }
          }
        }
      }
      return null;
    }
    public static String simple(String arg) {
      return arg;
    }

    public static String function(String arg) {
      return arg;
    }
  }

  private static class Pair {
    public int start;
    public int end;

    public Pair(int s) {
      start = s;
    }
  }
  private static String process_sql(String sql, Pair index) {
    String value = sql.substring(index.start, index.end);

    for (EscapeFunctions xx : EscapeFunctions.values()) {
      String newsql = xx.makeMatch(value);
      if (newsql != null) {
        sql = sql.substring(0, index.start) + newsql + " "
                + sql.substring(index.end + 1, sql.length());
        int x = newsql.length();

        index.end = index.start + newsql.length();

        break;
      }
    }
    return sql;
  }
  public static String parse(String sql) {
    Parser_return pR = parse(sql, 0);
    if (pR.bracket_cnt != 0) {
      throw new RuntimeException("Invalid java escape syntax - badly matched '}'");
    }
    return parse(sql, 0).sql_value;
  }
  static class Parser_return {
    public String sql_value;
    public int bracket_cnt;
  }
  private static Parser_return parse(String sql, int bracket_cnt) {
    int index = 0;
    boolean in_quote = false;

    do {
      if (sql.charAt(index) == '\'') {
        in_quote = !in_quote;
      } else if (sql.charAt(index) == '{' && !in_quote) {
        if (index + 1 == sql.length()) {
          // What ever string we get should have had limit nnn
          // added to the end and this test is veru unlikely
          throw new RuntimeException("Invalid java escape syntax - badly matched '{'");
        }
        Parser_return pR = parse(sql.substring(index + 1), ++bracket_cnt);
        bracket_cnt = pR.bracket_cnt;
        String sql_snippet = pR.sql_value;
        sql = sql.substring(0, index) + " " + sql_snippet;
      } else if (sql.charAt(index) == '}' && !in_quote) {
        Pair ptr = new Pair(0);
        ptr.end = index;
        Parser_return pR = new Parser_return();
        pR.sql_value = process_sql(sql, ptr);
        pR.bracket_cnt = --bracket_cnt;
        return pR;
      }
      index++;
    } while (index < sql.length());
    if (in_quote) {
      throw new RuntimeException("Invalid java escape syntax - badly matched '''");
    }

    Parser_return pR = new Parser_return();
    pR.sql_value = sql;
    pR.bracket_cnt = bracket_cnt;
    return pR;
  }
}

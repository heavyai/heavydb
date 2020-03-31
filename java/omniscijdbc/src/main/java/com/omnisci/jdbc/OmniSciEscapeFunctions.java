package com.omnisci.jdbc;

import java.lang.reflect.Method;
import java.sql.SQLException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public final class OmniSciEscapeFunctions {
  /**
   * storage for functions implementations
   */
  private static final ConcurrentMap<String, Method> FUNCTION_MAP =
          createFunctionMap("sql");

  private static ConcurrentMap<String, Method> createFunctionMap(String prefix) {
    Method[] methods = OmniSciEscapeFunctions.class.getMethods();
    ConcurrentMap<String, Method> functionMap =
            new ConcurrentHashMap<String, Method>(methods.length * 2);
    for (Method method : methods) {
      if (method.getName().startsWith(prefix)) {
        functionMap.put(
                method.getName().substring(prefix.length()).toLowerCase(Locale.US),
                method);
      }
    }
    return functionMap;
  }

  /**
   * get Method object implementing the given function
   *
   * @param functionName name of the searched function
   * @return a Method object or null if not found
   */
  public static Method getFunction(String functionName) {
    Method method = FUNCTION_MAP.get(functionName);
    if (method != null) {
      return method;
    }
    // FIXME: this probably should not use the US locale
    String nameLower = functionName.toLowerCase(Locale.US);
    if (nameLower.equals(functionName)) {
      // Input name was in lower case, the function is not there
      return null;
    }
    method = FUNCTION_MAP.get(nameLower);
    if (method != null && FUNCTION_MAP.size() < 1000) {
      // Avoid OutOfMemoryError in case input function names are randomized
      // The number of methods is finite, however the number of upper-lower case
      // combinations is quite a few (e.g. substr, Substr, sUbstr, SUbstr, etc).
      FUNCTION_MAP.putIfAbsent(functionName, method);
    }
    return method;
  }

  // ** numeric functions translations **

  /**
   * ceiling to ceil translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlceiling(StringBuilder buf,
          List<? extends CharSequence> parsedArgs) throws SQLException {
    singleArgumentFunctionCall(buf, "ceil(", "ceiling", parsedArgs);
  }

  /**
   * log to ln translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqllog(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "ln(", "log", parsedArgs);
  }

  /**
   * dayofmonth translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqldayofmonth(StringBuilder buf,
          List<? extends CharSequence> parsedArgs) throws SQLException {
    singleArgumentFunctionCall(buf, "extract(day from ", "dayofmonth", parsedArgs);
  }

  /**
   * dayofweek translation adding 1 to postgresql function since we expect values from 1
   * to 7
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqldayofweek(StringBuilder buf,
          List<? extends CharSequence> parsedArgs) throws SQLException {
    if (parsedArgs.size() != 1) {
      throw new RuntimeException(
              "Syntax error function 'dayofweek' takes one and only one argument.");
    }
    appendCall(buf, "extract(dow from ", ",", ")+1", parsedArgs);
  }

  /**
   * dayofyear translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqldayofyear(StringBuilder buf,
          List<? extends CharSequence> parsedArgs) throws SQLException {
    singleArgumentFunctionCall(buf, "extract(doy from ", "dayofyear", parsedArgs);
  }

  /**
   * hour translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlhour(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "extract(hour from ", "hour", parsedArgs);
  }

  /**
   * minute translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlminute(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "extract(minute from ", "minute", parsedArgs);
  }

  /**
   * month translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlmonth(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "extract(month from ", "month", parsedArgs);
  }

  /**
   * quarter translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlquarter(StringBuilder buf,
          List<? extends CharSequence> parsedArgs) throws SQLException {
    singleArgumentFunctionCall(buf, "extract(quarter from ", "quarter", parsedArgs);
  }

  /**
   * second translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlsecond(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "extract(second from ", "second", parsedArgs);
  }

  /**
   * week translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlweek(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "extract(week from ", "week", parsedArgs);
  }

  /**
   * year translation
   *
   * @param buf The buffer to append into
   * @param parsedArgs arguments
   * @throws SQLException if something wrong happens
   */
  public static void sqlyear(StringBuilder buf, List<? extends CharSequence> parsedArgs)
          throws SQLException {
    singleArgumentFunctionCall(buf, "extract(year from ", "year", parsedArgs);
  }

  private static void singleArgumentFunctionCall(StringBuilder buf,
          String call,
          String functionName,
          List<? extends CharSequence> parsedArgs) {
    if (parsedArgs.size() != 1) {
      throw new RuntimeException(
              "Syntax error " + functionName + " takes one and only one argument.");
    }
    CharSequence arg0 = parsedArgs.get(0);
    buf.ensureCapacity(buf.length() + call.length() + arg0.length() + 1);
    buf.append(call).append(arg0).append(')');
  }

  /**
   * Appends {@code begin arg0 separator arg1 separator end} sequence to the input {@link
   * StringBuilder}
   * @param sb destination StringBuilder
   * @param begin begin string
   * @param separator separator string
   * @param end end string
   * @param args arguments
   */

  public static void appendCall(StringBuilder sb,
          String begin,
          String separator,
          String end,
          List<? extends CharSequence> args) {
    int size = begin.length();
    // Typically just-in-time compiler would eliminate Iterator in case foreach is used,
    // however the code below uses indexed iteration to keep the conde independent from
    // various JIT implementations (== avoid Iterator allocations even for not-so-smart
    // JITs) see https://bugs.openjdk.java.net/browse/JDK-8166840 see
    // http://2016.jpoint.ru/talks/cheremin/ (video and slides)
    int numberOfArguments = args.size();
    for (int i = 0; i < numberOfArguments; i++) {
      size += args.get(i).length();
    }
    size += separator.length() * (numberOfArguments - 1);
    sb.ensureCapacity(sb.length() + size + 1);
    sb.append(begin);
    for (int i = 0; i < numberOfArguments; i++) {
      if (i > 0) {
        sb.append(separator);
      }
      sb.append(args.get(i));
    }
    sb.append(end);
  }
}

package com.mapd.tests;

import com.omnisci.thrift.server.TOmniSciException;
import com.omnisci.thrift.server.TTypeInfo;

import org.apache.commons.math3.util.Pair;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoField;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Random;
import java.util.function.Function;

/**
 *  a (hopefully) complete test case for date/time functions in OmniSci.
 */
public class DateTimeTest {
  static enum DateTruncUnit {
    dtYEAR("YEAR", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.withMonth(1);
        t = t.withDayOfMonth(1);
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    dtQUARTER("QUARTER", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        int month = t.getMonthValue();

        switch (month) {
          case 12:
          case 11:
          case 10:
            t = t.withMonth(10);
            break;
          case 9:
          case 8:
          case 7:
            t = t.withMonth(7);
            break;
          case 6:
          case 5:
          case 4:
            t = t.withMonth(4);
            break;
          case 3:
          case 2:
          case 1:
            t = t.withMonth(1);
            break;
        };

        t = t.withDayOfMonth(1);
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    dtMONTH("MONTH", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.withDayOfMonth(1);
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    dtDAY("DAY", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    dtHOUR("HOUR", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.HOURS);
        return t;
      }
    }),
    dtMINUTE("MINUTE", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.MINUTES);
        return t;
      }
    }),
    dtSECOND("SECOND", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.SECONDS);
        return t;
      }
    }),
    //		dtMILLENNIUM("MILLENNIUM", new Function<LocalDateTime, LocalDateTime>() {
    //			@Override
    //			public LocalDateTime apply(LocalDateTime t) {
    //				int year = t.getYear();
    //				int range = 1000;
    //				int diff = year % range;
    //				if (diff == 0) {
    //					diff = range;
    //				}
    //				year -= diff;
    //				t = t.withYear(year + 1);
    //				t = t.withMonth(1);
    //				t = t.withDayOfMonth(1);
    //				t = t.truncatedTo(ChronoUnit.DAYS);
    //				return t;
    //			}
    //		}),
    dtCENTURY("CENTURY", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        int year = t.getYear();
        int range = 100;
        int diff = year % range;
        if (diff == 0) {
          diff = range;
        }
        year -= diff;
        t = t.withYear(year + 1);

        t = t.withMonth(1);
        t = t.withDayOfMonth(1);
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    dtDECADE("DECADE", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        int year = t.getYear();
        int range = 10;
        int diff = year % range;
        year -= diff;
        t = t.withYear(year);
        t = t.withMonth(1);
        t = t.withDayOfMonth(1);
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    dtMILLISECOND("MILLISECOND", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.MILLIS);
        return t;
      }
    }),
    dtMICROSECOND("MICROSECOND", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.MICROS);
        return t;
      }
    }),
    dtNANOSECOND("NANOSECOND", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.truncatedTo(ChronoUnit.NANOS);
        return t;
      }
    }),
    dtWEEK("WEEK", new Function<LocalDateTime, LocalDateTime>() {
      @Override
      public LocalDateTime apply(LocalDateTime t) {
        t = t.with(ChronoField.DAY_OF_WEEK, 1);
        t = t.truncatedTo(ChronoUnit.DAYS);
        return t;
      }
    }),
    //		  dtQUARTERDAY("QUERTERDAY", new Function<LocalDateTime, LocalDateTime>()
    //{
    //			  @Override
    //			public LocalDateTime apply(LocalDateTime t) {
    //				  int hour = t.getHour();
    //				  hour /= 4;
    //
    //				  t = t.withHour(hour);
    //				t = t.truncatedTo(ChronoUnit.SECONDS);
    //				return t;
    //			}
    //		  })
    ;

    private String sqlToken;
    Function<LocalDateTime, LocalDateTime> trunc;

    private DateTruncUnit(String token, Function<LocalDateTime, LocalDateTime> trunc) {
      this.sqlToken = token;
      this.trunc = trunc;
    }
  }
  ;

  static enum DateExtractUnit {
    daYEAR("YEAR", new Function<LocalDateTime, Long>() {
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.YEAR);
      }
    }),
    daQUARTER("QUARTER", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        int month = t.get(ChronoField.MONTH_OF_YEAR);
        switch (month) {
          case 1:
          case 2:
          case 3:
            return 1l;
          case 4:
          case 5:
          case 6:
            return 2l;
          case 7:
          case 8:
          case 9:
            return 3l;
          case 10:
          case 11:
          case 12:
            return 4l;
        }
        return -1l;
      }
    }),
    daMONTH("MONTH", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.MONTH_OF_YEAR);
      }
    }),
    daDAY("DAY", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.DAY_OF_MONTH);
      }
    }),
    daHOUR("HOUR", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.HOUR_OF_DAY);
      }
    }),
    daMINUTE("MINUTE", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.MINUTE_OF_HOUR);
      }
    }),
    daSECOND("SECOND", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.SECOND_OF_MINUTE);
      }
    }),
    //		daMILLENNIUM("MILLENNIUM", ChronoField.YEAR, 1000),
    //		daCENTURY("CENTURY", ChronoField.YEAR, 100),
    //		daDECADE("DECADE", ChronoField.YEAR, 10),
    daMILLISECOND("MILLISECOND", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return t.get(ChronoField.MILLI_OF_SECOND)
                + (1000L * t.get(ChronoField.SECOND_OF_MINUTE));
      }
    }),
    daMICROSECOND("MICROSECOND", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return t.get(ChronoField.MICRO_OF_SECOND)
                + (1000_000L * t.get(ChronoField.SECOND_OF_MINUTE));
      }
    }),
    daNANOSECOND("NANOSECOND", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return t.get(ChronoField.NANO_OF_SECOND)
                + (1000_000_000L * t.get(ChronoField.SECOND_OF_MINUTE));
      }
    }),
    daWEEK("WEEK", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        LocalDateTime year = DateTruncUnit.dtYEAR.trunc.apply(t);
        // bring it to the 4th of Jan (as this is always in the first week of the year)
        year = year.plusDays(3);

        // compute the start day of that week (the Monday)
        LocalDateTime week = DateTruncUnit.dtWEEK.trunc.apply(year);

        if (week.compareTo(t) > 0) {
          year = year.minusYears(1);
          week = DateTruncUnit.dtWEEK.trunc.apply(year);
        }

        int weeks = 0;
        while (week.compareTo(t) <= 0) {
          weeks++;
          week = week.plusWeeks(1);
        }

        return (long) weeks;
      }
    }),
    //		daQUARTERDAY("QUERTERDAY", new Function<LocalDateTime, Integer>() {
    //			@Override
    //			public Integer apply(LocalDateTime t) {
    //				return ((t.get(ChronoField.HOUR_OF_DAY) -1 ) / 4) + 1;
    //			}
    //		}),
    //		daWEEKDAY("WEEKDAYS", new Function<LocalDateTime, Integer>() {
    //			@Override
    //			public Integer apply(LocalDateTime t) {
    //				return t.get(ChronoField.DAY_OF_WEEK);
    //			}
    //		}),
    daDAYOFYEAR("DOY", new Function<LocalDateTime, Long>() {
      @Override
      public Long apply(LocalDateTime t) {
        return (long) t.get(ChronoField.DAY_OF_YEAR);
      }
    });

    private String sqlToken;
    private Function<LocalDateTime, Long> extract;

    private DateExtractUnit(String token, Function<LocalDateTime, Long> f) {
      this.sqlToken = token;
      this.extract = f;
    }
  }
  ;

  static enum DateDiffUnit {
    daYEAR("YEAR", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.YEARS);
      }
    }),
    daQUARTER("QUARTER", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      private Long applyCorrect(Pair<LocalDateTime, LocalDateTime> d) {
        LocalDateTime start = d.getFirst();
        LocalDateTime end = d.getSecond();

        int delta = 1;
        if (start.compareTo(end) > 0) {
          delta = -1;
          start = end;
          end = d.getFirst();
        }

        start = DateTruncUnit.dtQUARTER.trunc.apply(start);
        //				end = DateTruncUnit.dtQUARTER.trunc.apply(end);
        long rc = 0;

        while (start.compareTo(end) <= 0) {
          rc += delta;
          start = start.plusMonths(3);
        }

        return rc;
      }

      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        // this seems to be what mysql does
        return d.getFirst().until(d.getSecond(), ChronoUnit.MONTHS) / 3;
      }
    }),
    daMONTH("MONTH", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.MONTHS);
      }
    }),
    daDAY("DAY", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.DAYS);
      }
    }),
    daHOUR("HOUR", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.HOURS);
      }
    }),
    daMINUTE("MINUTE", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.MINUTES);
      }
    }),
    daSECOND("SECOND", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.SECONDS);
      }
    }),
    //		daMILLENNIUM("MILLENNIUM", ChronoField.YEAR, 1000),
    //		daCENTURY("CENTURY", ChronoField.YEAR, 100),
    //		daDECADE("DECADE", ChronoField.YEAR, 10),
    daMILLISECOND(
            "MILLISECOND", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
              public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
                return d.getFirst().until(d.getSecond(), ChronoUnit.MILLIS);
              }
            }),
    daMICROSECOND(
            "MICROSECOND", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
              public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
                return d.getFirst().until(d.getSecond(), ChronoUnit.MICROS);
              }
            }),
    daNANOSECOND("NANOSECOND", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.NANOS);
      }
    }),
    daWEEK("WEEK", new Function<Pair<LocalDateTime, LocalDateTime>, Long>() {
      public Long apply(Pair<LocalDateTime, LocalDateTime> d) {
        return d.getFirst().until(d.getSecond(), ChronoUnit.WEEKS);
      }
    }),
    //		daQUARTERDAY("QUERTERDAY", new Function<LocalDateTime, Integer>() {
    //			@Override
    //			public Integer apply(LocalDateTime t) {
    //				return ((t.get(ChronoField.HOUR_OF_DAY) -1 ) / 4) + 1;
    //			}
    //		}),
    //		daWEEKDAY("WEEKDAYS", new Function<LocalDateTime, Integer>() {
    //			@Override
    //			public Integer apply(LocalDateTime t) {
    //				return t.get(ChronoField.DAY_OF_WEEK);
    //			}
    //		}),
    ;

    private String sqlToken;
    private Function<Pair<LocalDateTime, LocalDateTime>, Long> diff;

    private DateDiffUnit(
            String token, Function<Pair<LocalDateTime, LocalDateTime>, Long> diff) {
      this.sqlToken = token;
      this.diff = diff;
    }
  }
  ;

  static enum DateAddUnit {
    daYEAR("YEAR", 99, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.YEARS);
      }
    }),
    daQUARTER(
            "QUARTER", 10 * 3, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
              @Override
              public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
                return t.getFirst().plus(t.getSecond() * 3, ChronoUnit.MONTHS);
              }
            }),
    daMONTH("MONTH", 99, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      @Override
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.MONTHS);
      }
    }),
    daDAY("DAY", 99, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      @Override
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.DAYS);
      }
    }),
    daHOUR("HOUR", 99, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      @Override
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.HOURS);
      }
    }),
    daMINUTE("MINUTE", 99, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      @Override
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.MINUTES);
      }
    }),
    daSECOND("SECOND", 99, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      @Override
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.SECONDS);
      }
    }),
    //		daMILLENNIUM("MILLENNIUM", ChronoField.YEAR, 1000),
    //		daCENTURY("CENTURY", ChronoField.YEAR, 100),
    //		daDECADE("DECADE", ChronoField.YEAR, 10),
    daMILLISECOND("MILLISECOND",
            12 * 30 * 24 * 60 * 60 * 1000L,
            new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
              @Override
              public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
                return t.getFirst().plus(t.getSecond(), ChronoUnit.MILLIS);
              }
            }),
    daMICROSECOND("MICROSECOND",
            12 * 30 * 24 * 60 * 60 * 1000 * 1000,
            new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
              @Override
              public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
                return t.getFirst().plus(t.getSecond(), ChronoUnit.MICROS);
              }
            }),
    daNANOSECOND("NANOSECOND",
            12 * 30 * 24 * 60 * 60 * 1000 * 1000 * 1000,
            new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
              @Override
              public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
                return t.getFirst().plus(t.getSecond(), ChronoUnit.NANOS);
              }
            }),
    daWEEK("WEEK", 53, new Function<Pair<LocalDateTime, Long>, LocalDateTime>() {
      @Override
      public LocalDateTime apply(Pair<LocalDateTime, Long> t) {
        return t.getFirst().plus(t.getSecond(), ChronoUnit.WEEKS);
      }
    }),
    //		daQUARTERDAY("QUERTERDAY", new Function<LocalDateTime, Integer>() {
    //			@Override
    //			public Integer apply(LocalDateTime t) {
    //				return ((t.get(ChronoField.HOUR_OF_DAY) -1 ) / 4) + 1;
    //			}
    //		}),
    //		daWEEKDAY("WEEKDAYS", new Function<LocalDateTime, Integer>() {
    //			@Override
    //			public Integer apply(LocalDateTime t) {
    //				return t.get(ChronoField.DAY_OF_WEEK);
    //			}
    //		}),
    ;

    private String sqlToken;
    private Function<Pair<LocalDateTime, Long>, LocalDateTime> add;
    private long max;

    private DateAddUnit(String token,
            long max,
            Function<Pair<LocalDateTime, Long>, LocalDateTime> f) {
      this.sqlToken = token;
      this.max = max;
      this.add = f;
    }
  }
  ;

  static LocalDateTime createRandomDateTime(Random r) {
    try {
      int year = 1900 + r.nextInt(200);
      int month = 1 + r.nextInt(12);
      int dayOfMonth = 1 + r.nextInt(31);
      int hour = r.nextInt(24);
      int minute = r.nextInt(60);
      int second = r.nextInt(60);
      int nanoOfSecond = r.nextInt(1000 * 1000 * 1000);

      return LocalDateTime.of(
              year, month, dayOfMonth, hour, minute, second, nanoOfSecond);
    } catch (Exception e) {
      return createRandomDateTime(r);
    }
  }

  static enum Encoding {
    TIMESTAMP("TIMESTAMP", "'TIMESTAMP' ''yyyy-MM-dd HH:mm:ss''", ChronoUnit.SECONDS),
    TIMESTAMP_0(
            "TIMESTAMP(0)", "'TIMESTAMP(0)' ''yyyy-MM-dd HH:mm:ss''", ChronoUnit.SECONDS),
    TIMESTAMP_3("TIMESTAMP(3)",
            "'TIMESTAMP(3)' ''yyyy-MM-dd HH:mm:ss.SSS''",
            ChronoUnit.MILLIS),
    TIMESTAMP_6("TIMESTAMP(6)",
            "'TIMESTAMP(6)' ''yyyy-MM-dd HH:mm:ss.SSSSSS''",
            ChronoUnit.MICROS),
    TIMESTAMP_9("TIMESTAMP(9)",
            "'TIMESTAMP(9)' ''yyyy-MM-dd HH:mm:ss.SSSSSSSSS''",
            ChronoUnit.NANOS),
    TIMESTAMP_FIXED_32("TIMESTAMP ENCODING FIXED(32)",
            "'TIMESTAMP' ''yyyy-MM-dd HH:mm:ss''",
            ChronoUnit.SECONDS),
    DATE("DATE", "'DATE' ''yyyy-MM-dd''", ChronoUnit.DAYS),
    DATE_FIXED_16("DATE ENCODING FIXED(16)", "'DATE' ''yyyy-MM-dd''", ChronoUnit.DAYS),
    DATE_FIXED_32("DATE ENCODING FIXED(32)", "'DATE' ''yyyy-MM-dd''", ChronoUnit.DAYS);

    DateTimeFormatter formatter;
    String sqlType;
    ChronoUnit toClear;

    Encoding(String sqlType, String pattern, ChronoUnit unit) {
      this.sqlType = sqlType;
      formatter = DateTimeFormatter.ofPattern(pattern);
      this.toClear = unit;
    }

    public String toSqlColumn(String prefx, LocalDateTime val) {
      if (null != val) return prefx + "_" + name() + " /* " + toSql(val) + " */";
      return prefx + "_" + name();
    }

    public String toSql(LocalDateTime d) {
      return formatter.format(d);
    }

    public LocalDateTime clear(LocalDateTime d) {
      if (null != toClear) {
        d = d.truncatedTo(toClear);
      }

      return d;
    }

    public LocalDateTime clearForDateAddResult(LocalDateTime d) {
      if (null != toClear) {
        if (toClear == ChronoUnit.DAYS) {
          d = d.truncatedTo(ChronoUnit.SECONDS);
        } else {
          d = d.truncatedTo(toClear);
        }
      }

      return d;
    }
  }

  static LocalDateTime getDateTimeFromQuery(MapdTestClient client, String sql)
          throws Exception {
    try {
      com.omnisci.thrift.server.TQueryResult res = client.runSql(sql);
      LocalDateTime r = null;
      if (res.row_set.is_columnar) {
        TTypeInfo tt = res.row_set.row_desc.get(0).col_type;
        int pow = (int) Math.pow(10, tt.precision);
        long val = res.row_set.columns.get(0).data.int_col.get(0);
        int nanosPow = (int) Math.pow(10, 9 - tt.precision);
        long nanos = (val % pow);
        if (nanos < 0) {
          nanos = pow + nanos;
        }
        nanos *= nanosPow;
        r = LocalDateTime.ofEpochSecond(
                Math.floorDiv(val, pow), (int) nanos, ZoneOffset.UTC);

      } else {
        throw new RuntimeException("Unsupported!");
      }

      return r;
    } catch (TOmniSciException e) {
      System.out.println("Query failed: " + sql + " -- " + e.error_msg);
      return LocalDateTime.MIN;

    } catch (Exception e) {
      System.out.println("Query failed: " + sql + " -- " + e.getMessage());
      return LocalDateTime.MIN;
    }
  }

  static long getLongFromQuery(MapdTestClient client, String sql) throws Exception {
    try {
      com.omnisci.thrift.server.TQueryResult res = client.runSql(sql);
      long r = -1;
      if (res.row_set.is_columnar) {
        long val = res.row_set.columns.get(0).data.int_col.get(0);
        r = val;
      } else {
        throw new RuntimeException("Unsupported!");
      }
      return r;
    } catch (TOmniSciException e) {
      System.out.println("Query failed: " + sql + " -- " + e.error_msg);
      return Long.MIN_VALUE;
    } catch (Exception e) {
      System.out.println("Query failed: " + sql + " -- " + e.getMessage());
      return Long.MIN_VALUE;
    }
  }

  public static LocalDateTime testDateTrunc(
          LocalDateTime d, DateTruncUnit f, MapdTestClient client, Encoding enc)
          throws Exception {
    String sql = "SELECT DATE_TRUNC('" + f.sqlToken + "', " + enc.toSql(d) + ");";
    LocalDateTime r = getDateTimeFromQuery(client, sql);
    LocalDateTime expected = f.trunc.apply(d);
    expected = enc.clear(expected);

    Fuzzy rc = Fuzzy.compare(expected, r, enc);
    if (resultsToDump.contains(rc)) {
      System.out.println("Query " + rc + ": " + sql
              + " -> expected: " + expected.toString() + " got " + r.toString());
    }

    return testDateTruncTable(d, f, client, enc);
  }

  private static void updateValues(MapdTestClient client, LocalDateTime a, Encoding aEnc)
          throws Exception {
    updateValues(client, a, aEnc, null, null);
  }

  private static void updateValues(MapdTestClient client,
          LocalDateTime a,
          Encoding aEnc,
          LocalDateTime b,
          Encoding bEnc) throws Exception {
    String sqlUpdate = "UPDATE DateTimeTest set " + aEnc.toSqlColumn("a", null) + " = "
            + aEnc.toSql(a);

    if (null != b) {
      sqlUpdate += ", " + bEnc.toSqlColumn("b", null) + " = " + bEnc.toSql(b);
    }

    sqlUpdate += ";";

    try {
      client.runSql(sqlUpdate);
    } catch (TOmniSciException e) {
      System.out.println("Update failed: " + sqlUpdate + " " + e.error_msg);
    }
  }

  public static LocalDateTime testDateTruncTable(
          LocalDateTime d, DateTruncUnit f, MapdTestClient client, Encoding enc)
          throws Exception {
    updateValues(client, d, enc);
    String sql = "SELECT DATE_TRUNC('" + f.sqlToken + "', " + enc.toSqlColumn("a", d)
            + ") FROM DateTimeTest;";
    LocalDateTime r = getDateTimeFromQuery(client, sql);
    LocalDateTime expected = f.trunc.apply(d);
    expected = enc.clear(expected);

    Fuzzy rc = Fuzzy.compare(expected, r, enc);
    if (resultsToDump.contains(rc)) {
      System.out.println("Query " + rc + ": " + sql
              + " -> expected: " + expected.toString() + " got " + r.toString());
    }

    return expected;
  }

  public static void testDateExtract(
          LocalDateTime d, DateExtractUnit f, MapdTestClient client, Encoding enc)
          throws Exception {
    String sql = "SELECT EXTRACT(" + f.sqlToken + " FROM " + enc.toSql(d) + ");";
    long r = getLongFromQuery(client, sql);

    d = enc.clear(d);
    long expected = f.extract.apply(d);

    Fuzzy rc = Fuzzy.compare(expected, r);
    if (resultsToDump.contains(rc)) {
      System.out.println(
              "Query " + rc + ": " + sql + " -> expected: " + expected + " got " + r);
    }

    testDateExtractTable(d, f, client, enc);
  }

  public static void testDateExtractTable(
          LocalDateTime d, DateExtractUnit f, MapdTestClient client, Encoding enc)
          throws Exception {
    updateValues(client, d, enc);
    String sql = "SELECT EXTRACT(" + f.sqlToken + " FROM " + enc.toSqlColumn("a", d)
            + ") FROM DateTimeTest;";
    long r = getLongFromQuery(client, sql);

    d = enc.clear(d);
    long expected = f.extract.apply(d);

    Fuzzy rc = Fuzzy.compare(expected, r);
    if (resultsToDump.contains(rc)) {
      System.out.println(
              "Query " + rc + ": " + sql + " -> expected: " + expected + " got " + r);
    }
  }

  public static void testDiff(String fn,
          LocalDateTime d0,
          LocalDateTime d1,
          DateDiffUnit f,
          MapdTestClient client,
          Encoding enc0,
          Encoding enc1) throws Exception {
    String sql = "SELECT " + fn + "(" + f.sqlToken + ", " + enc0.toSql(d0) + ", "
            + enc1.toSql(d1) + ");";
    long r = getLongFromQuery(client, sql);
    d0 = enc0.clear(d0);
    d1 = enc1.clear(d1);

    long expected = f.diff.apply(Pair.create(d0, d1));

    Fuzzy rc = Fuzzy.compare(expected, r);
    if (resultsToDump.contains(rc)) {
      System.out.println(
              "Query " + rc + ": " + sql + " -> expected: " + expected + " got " + r);
    }

    testDiffTable(fn, d0, d1, f, client, enc0, enc1);
  }

  public static void testDiffTable(String fn,
          LocalDateTime d0,
          LocalDateTime d1,
          DateDiffUnit f,
          MapdTestClient client,
          Encoding enc0,
          Encoding enc1) throws Exception {
    updateValues(client, d0, enc0, d1, enc1);
    String sql = "SELECT " + fn + "(" + f.sqlToken + ", " + enc0.toSqlColumn("a", d0)
            + ", " + enc1.toSqlColumn("b", d1) + ") FROM DateTimeTest;";
    long r = getLongFromQuery(client, sql);
    d0 = enc0.clear(d0);
    d1 = enc1.clear(d1);

    long expected = f.diff.apply(Pair.create(d0, d1));

    Fuzzy rc = Fuzzy.compare(expected, r);
    if (resultsToDump.contains(rc)) {
      System.out.println(
              "Query " + rc + ": " + sql + " -> expected: " + expected + " got " + r);
    }
  }

  public static void testDateAdd(String fn,
          LocalDateTime d,
          DateAddUnit f,
          long units,
          MapdTestClient client,
          Encoding enc) throws Exception {
    String sql =
            "SELECT " + fn + "(" + f.sqlToken + ", " + units + ", " + enc.toSql(d) + ");";
    LocalDateTime r = getDateTimeFromQuery(client, sql);

    LocalDateTime expected = f.add.apply(Pair.create(enc.clear(d), units));
    expected = enc.clearForDateAddResult(expected);

    Fuzzy rc = Fuzzy.compareDateAdd(expected, r, enc);
    if (resultsToDump.contains(rc)) {
      System.out.println("Query " + rc + ": " + sql
              + " -> expected: " + expected.toString() + " got " + r.toString());
    }

    testDateAddTable(fn, d, f, units, client, enc);
  }

  public static void testDateAddTable(String fn,
          LocalDateTime d,
          DateAddUnit f,
          long units,
          MapdTestClient client,
          Encoding enc) throws Exception {
    updateValues(client, d, enc);
    String sql = "SELECT " + fn + "(" + f.sqlToken + ", " + units + ", "
            + enc.toSqlColumn("a", d) + ") FROM DateTimeTest;";
    LocalDateTime r = getDateTimeFromQuery(client, sql);

    LocalDateTime expected = f.add.apply(Pair.create(enc.clear(d), units));
    expected = enc.clearForDateAddResult(expected);

    Fuzzy rc = Fuzzy.compareDateAdd(expected, r, enc);
    if (resultsToDump.contains(rc)) {
      System.out.println("Query " + rc + ": " + sql
              + " -> expected: " + expected.toString() + " got " + r.toString());
    }
  }

  static EnumSet resultsToDump = EnumSet.of(Fuzzy.failed, Fuzzy.okish);

  static EnumSet addAllowed = EnumSet.allOf(DateAddUnit.class);

  static {
    addAllowed.remove(DateAddUnit.daQUARTER);
    addAllowed.remove(DateAddUnit.daMILLISECOND);
    addAllowed.remove(DateAddUnit.daMICROSECOND);
    addAllowed.remove(DateAddUnit.daNANOSECOND);
    addAllowed.remove(DateAddUnit.daWEEK);
  }

  public static void testAdd(
          LocalDateTime d, DateAddUnit f, long units, MapdTestClient client, Encoding enc)
          throws Exception {
    if (!addAllowed.contains(f)) {
      return;
    }

    String sql =
            "SELECT " + enc.toSql(d) + " + INTERVAL '" + units + "' " + f.sqlToken + " ;";
    LocalDateTime r = getDateTimeFromQuery(client, sql);

    LocalDateTime expected = f.add.apply(Pair.create(enc.clear(d), units));
    expected = enc.clearForDateAddResult(expected);

    Fuzzy rc = Fuzzy.compareDateAdd(expected, r, enc);
    if (resultsToDump.contains(rc)) {
      System.out.println("Query " + rc + ": " + sql
              + " -> expected: " + expected.toString() + " got " + r.toString());
    }
  }

  public static void testSub(
          LocalDateTime d, DateAddUnit f, long units, MapdTestClient client, Encoding enc)
          throws Exception {
    if (!addAllowed.contains(f)) {
      return;
    }

    long toSub = -units;

    String sql =
            "SELECT " + enc.toSql(d) + " - INTERVAL '" + toSub + "' " + f.sqlToken + " ;";
    LocalDateTime r = getDateTimeFromQuery(client, sql);

    LocalDateTime expected = f.add.apply(Pair.create(enc.clear(d), units));
    expected = enc.clearForDateAddResult(expected);

    Fuzzy rc = Fuzzy.compareDateAdd(expected, r, enc);
    if (resultsToDump.contains(rc)) {
      System.out.println("Query " + rc + ": " + sql
              + " -> expected: " + expected.toString() + " got " + r.toString());
    }
  }

  static enum Fuzzy {
    ok,
    okish,
    failed;

    static Fuzzy compare(LocalDateTime expected, LocalDateTime result, Encoding enc) {
      if (expected.equals(result)) return ok;

      LocalDateTime okish = result.minus(1, ChronoUnit.NANOS);
      okish = enc.clear(okish);

      if (expected.equals(okish)) return Fuzzy.okish;

      okish = result.plus(1, ChronoUnit.NANOS);
      okish = enc.clear(okish);

      if (expected.equals(okish)) return Fuzzy.okish;

      return failed;
    }

    static Fuzzy compare(long expected, long result) {
      if (expected == result) return ok;

      long okish = result - 1;

      if (expected == okish) return Fuzzy.okish;

      okish = result + 1;

      if (expected == okish) return Fuzzy.okish;

      return failed;
    }

    static Fuzzy compareDateAdd(
            LocalDateTime expected, LocalDateTime result, Encoding enc) {
      if (expected.equals(result)) return ok;

      LocalDateTime okish = result.minus(1, ChronoUnit.NANOS);
      okish = enc.clearForDateAddResult(okish);

      if (expected.equals(okish)) return Fuzzy.okish;

      okish = result.plus(1, ChronoUnit.NANOS);
      okish = enc.clearForDateAddResult(okish);

      if (expected.equals(okish)) return Fuzzy.okish;

      return failed;
    }
  }

  public static void createTestTable(MapdTestClient client) throws Exception {
    client.runSql("DROP TABLE IF EXISTS DateTimeTest;");
    String sqlCreate = "CREATE TABLE DateTimeTest(id int";
    String sqlInsert = "INSERT INTO DateTimeTest VALUES(0";
    for (Encoding e : Encoding.values()) {
      sqlCreate += ", " + e.toSqlColumn("a", null) + " " + e.sqlType;
      sqlCreate += ", " + e.toSqlColumn("b", null) + " " + e.sqlType;
      sqlInsert += ", null, null";
    }

    sqlCreate += ");";
    sqlInsert += ");";

    client.runSql(sqlCreate);
    client.runSql(sqlInsert);

    System.out.println("CREATE: " + sqlCreate);
    System.out.println("INSERT: " + sqlInsert);
  }

  public static void main(String[] args) throws Exception {
    long seed = System.currentTimeMillis();

    // to reproduce a previous run, use the same seed
    // seed = 1593081011125L;

    System.out.println("Seed: " + seed);
    Random r = new Random(seed);

    MapdTestClient su = MapdTestClient.getClient(
            "localhost", 6274, "omnisci", "admin", "HyperInteractive");
    LocalDateTime d0 = createRandomDateTime(r);
    LocalDateTime d1 = createRandomDateTime(r);

    createTestTable(su);

    // don't dump OK results
    resultsToDump = EnumSet.of(Fuzzy.failed, Fuzzy.okish);
    boolean testTrunc = true;
    boolean testExtract = true;
    boolean testDiff = true;
    boolean testAdd = true;

    if (testTrunc) {
      for (Encoding enc0 : Encoding.values()) {
        for (DateTruncUnit f : DateTruncUnit.values()) {
          LocalDateTime e = testDateTrunc(d0, f, su, enc0);
          e = e.minus(1, ChronoUnit.NANOS);
          testDateTrunc(e, f, su, enc0);

          e = testDateTrunc(d1, f, su, enc0);
          e = e.minus(1, ChronoUnit.NANOS);
          testDateTrunc(e, f, su, enc0);
        }
      }
    }

    if (testExtract) {
      for (Encoding enc0 : Encoding.values()) {
        for (DateExtractUnit f : DateExtractUnit.values()) {
          testDateExtract(d0, f, su, enc0);
          testDateExtract(d0.minusNanos(1), f, su, enc0);
          testDateExtract(d0.plusNanos(1), f, su, enc0);
          testDateExtract(d1, f, su, enc0);
          testDateExtract(d1.minusNanos(1), f, su, enc0);
          testDateExtract(d1.plusNanos(1), f, su, enc0);
        }
      }
    }

    if (testDiff) {
      for (Encoding enc0 : Encoding.values()) {
        for (Encoding enc1 : Encoding.values()) {
          for (DateDiffUnit f : DateDiffUnit.values()) {
            for (String fn : Arrays.asList("TIMESTAMPDIFF" /* , "DATEDIFF" */)) {
              testDiff(fn, d0, d1, f, su, enc0, enc1);
              testDiff(fn, d1, d0, f, su, enc0, enc1);
              testDiff(fn, d0, d0, f, su, enc0, enc1);
              testDiff(fn, d1, d1, f, su, enc0, enc1);
            }
          }
        }
      }
    }

    if (testAdd) {
      for (DateAddUnit f : DateAddUnit.values()) {
        long units = r.nextLong() % f.max;
        if (r.nextBoolean()) {
          units *= -1L;
        }
        for (Encoding enc0 : Encoding.values()) {
          for (String fn : Arrays.asList("TIMESTAMPADD", "DATEADD")) {
            testDateAdd(fn, d0, f, units, su, enc0);
            testDateAdd(fn, d1, f, units, su, enc0);
          }
          testAdd(d0, f, units, su, enc0);
          testSub(d0, f, units, su, enc0);
          testAdd(d1, f, units, su, enc0);
          testSub(d1, f, units, su, enc0);
        }
      }
    }
  }
}

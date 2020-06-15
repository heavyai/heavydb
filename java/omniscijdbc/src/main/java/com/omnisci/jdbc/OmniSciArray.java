package com.omnisci.jdbc;

import static java.lang.Math.toIntExact;

import com.omnisci.thrift.server.*;

import java.math.BigDecimal;
import java.sql.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Map;

public class OmniSciArray implements java.sql.Array {
  private TDatumType type;
  private Object[] elements;

  public OmniSciArray(TDatumType type, Object[] elements) throws SQLException {
    if (elements == null) {
      throw new SQLException("Elements[] cannot be null");
    }
    this.type = type;
    this.elements = elements;
    Class<?> elements_class = elements.getClass().getComponentType();
    switch (type) {
      case TINYINT:
        checkClass(elements_class, Byte.class);
        break;
      case SMALLINT:
        checkClass(elements_class, Short.class);
        break;
      case INT:
        checkClass(elements_class, Integer.class);
        break;
      case BIGINT:
        checkClass(elements_class, Long.class);
        break;
      case BOOL:
        checkClass(elements_class, Boolean.class);
        break;
      case TIME:
        checkClass(elements_class, java.sql.Time.class);
        break;
      case TIMESTAMP:
        checkClass(elements_class, java.sql.Timestamp.class);
        break;
      case DATE:
        checkClass(elements_class, java.sql.Date.class);
        break;
      case FLOAT:
        checkClass(elements_class, Float.class);
        break;
      case DECIMAL:
        checkClass(elements_class, BigDecimal.class);
        break;
      case DOUBLE:
        checkClass(elements_class, Double.class);
        break;
      case STR:
      case POINT:
      case LINESTRING:
      case POLYGON:
      case MULTIPOLYGON:
        checkClass(elements_class, String.class);
        break;
      default:
        throw new AssertionError(type.toString());
    }
  }

  @Override
  public String getBaseTypeName() throws SQLException {
    return type.name();
  }

  @Override
  public int getBaseType() throws SQLException {
    return OmniSciType.toJava(type);
  }

  @Override
  public Object getArray() throws SQLException {
    return elements;
  }

  @Override
  public Object getArray(long start, int size) throws SQLException {
    checkSize(toIntExact(start), size);
    return Arrays.copyOfRange(elements, toIntExact(start), size);
  }

  @Override
  public ResultSet getResultSet() throws SQLException {
    return getResultSet(0, elements.length);
  }

  @Override
  public ResultSet getResultSet(long start, int size) throws SQLException {
    checkSize(toIntExact(start), size);

    ArrayList<TColumnType> columnTypes = new ArrayList<>(2);
    TTypeInfo idxType =
            new TTypeInfo(TDatumType.BIGINT, TEncodingType.NONE, false, false, 0, 0, 0);
    columnTypes.add(new TColumnType("INDEX", idxType, false, "", false, false, 0));
    // @NOTE(Max): The problem here is that it's hard to know precision and scale.
    // But it looks like we don't use those anywhere in ResultSet ???
    int precision = (type == TDatumType.TIMESTAMP || type == TDatumType.TIME
                            || type == TDatumType.DATE
                    ? 3
                    : 0);
    TTypeInfo valueType =
            new TTypeInfo(type, TEncodingType.NONE, true, false, precision, 0, 0);
    columnTypes.add(new TColumnType("VALUE", valueType, false, "", false, false, 1));

    Long[] indexes = new Long[size];
    // indexes in SQL arrays start from 1
    for (int i = 0; i < size; ++i) {
      indexes[i] = (long) (i + 1);
    }
    TColumnData idxData = new TColumnData(Arrays.asList(indexes), null, null, null);
    ArrayList<Boolean> idxNulls = new ArrayList<>(size);
    for (int i = 0; i < size; ++i) {
      idxNulls.add(Boolean.FALSE);
    }

    TColumnData valuesData;
    Long[] int_values = new Long[size];
    Double[] real_values = new Double[size];
    String[] string_values = new String[size];
    boolean is_real = false, is_string = false;
    ArrayList<Boolean> valueNulls = new ArrayList<>(size);
    for (int i = toIntExact(start); i < start + size; ++i) {
      if (elements[i] == null) {
        valueNulls.add(true);
      } else {
        valueNulls.add(false);
        switch (type) {
          case TINYINT:
            int_values[i] = ((Byte) elements[i]).longValue();
            break;
          case SMALLINT:
            int_values[i] = ((Short) elements[i]).longValue();
            break;
          case INT:
            int_values[i] = ((Integer) elements[i]).longValue();
            break;
          case BIGINT:
            int_values[i] = (Long) elements[i];
            break;
          case BOOL:
            int_values[i] = elements[i] == Boolean.TRUE ? 1l : 0l;
            break;
          case TIME:
            int_values[i] = ((Time) elements[i]).getTime();
            break;
          case TIMESTAMP:
            int_values[i] = ((Timestamp) elements[i]).getTime();
            break;
          case DATE:
            int_values[i] = ((Date) elements[i]).getTime();
            break;
          case FLOAT:
            is_real = true;
            real_values[i] = ((Float) elements[i]).doubleValue();
            break;
          case DECIMAL:
            is_real = true;
            real_values[i] = ((BigDecimal) elements[i]).doubleValue();
            break;
          case DOUBLE:
            is_real = true;
            real_values[i] = (Double) elements[i];
            break;
          case STR:
          case POINT:
          case LINESTRING:
          case POLYGON:
          case MULTIPOLYGON:
            is_string = true;
            string_values[i] = (String) elements[i];
            break;
          default:
            throw new AssertionError(type.toString());
        }
      }
    }
    if (is_real) {
      valuesData = new TColumnData(null, Arrays.asList(real_values), null, null);
    } else if (is_string) {
      valuesData = new TColumnData(null, null, Arrays.asList(string_values), null);
    } else {
      valuesData = new TColumnData(Arrays.asList(int_values), null, null, null);
    }

    ArrayList<TColumn> columns = new ArrayList<>(2);
    columns.add(new TColumn(idxData, idxNulls));
    columns.add(new TColumn(valuesData, valueNulls));
    TRowSet rowSet = new TRowSet(columnTypes, null, columns, true);
    TQueryResult result = new TQueryResult(rowSet, 0, 0, "", "", true, TQueryType.READ);
    return new OmniSciResultSet(result, "");
  }

  @Override
  public void free() throws SQLException {
    elements = null;
  }

  @Override
  public String toString() {
    if (elements == null) {
      return "NULL";
    } else {
      switch (type) {
        case STR:
        case POINT:
        case LINESTRING:
        case POLYGON:
        case MULTIPOLYGON:
        case TIME:
        case TIMESTAMP:
        case DATE: {
          StringBuilder sb = new StringBuilder("{");
          for (Object e : elements) {
            if (e != null) {
              sb.append("'").append(e.toString()).append("', ");
            } else {
              sb.append("NULL").append(", ");
            }
          }
          if (elements.length > 0) {
            sb.delete(sb.length() - 2, sb.length());
          }
          sb.append("}");
          return sb.toString();
        }
        default: {
          String arr_str = Arrays.toString(elements);
          return "{" + arr_str.substring(1, arr_str.length() - 1) + "}";
        }
      }
    }
  }

  @Override
  public ResultSet getResultSet(long start, int size, Map<String, Class<?>> map)
          throws SQLException {
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public ResultSet getResultSet(Map<String, Class<?>> map) throws SQLException {
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Object getArray(long start, int size, Map<String, Class<?>> map)
          throws SQLException {
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  @Override
  public Object getArray(Map<String, Class<?>> map) throws SQLException {
    throw new UnsupportedOperationException("Not supported yet,"
            + " line:" + new Throwable().getStackTrace()[0].getLineNumber()
            + " class:" + new Throwable().getStackTrace()[0].getClassName()
            + " method:" + new Throwable().getStackTrace()[0].getMethodName());
  }

  private void checkSize(int start, int size) throws SQLException {
    if (start < 0 || start >= elements.length || start + size > elements.length) {
      throw new SQLException("Array length = " + Integer.toString(elements.length)
              + ", slice start index = " + Integer.toString(start)
              + ", slice length = " + Integer.toString(size));
    }
  }

  private void checkClass(Class<?> given, Class<?> expected) throws SQLException {
    if (!expected.isAssignableFrom(given)) {
      throw new SQLException("For array of " + getBaseTypeName() + ", elements of type "
              + expected + " are expected. Got " + given + " instead");
    }
  }
}

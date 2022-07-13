package com.mapd.utility.db_vendors;

import org.postgis.PGgeometry;
import org.postgresql.geometric.PGlseg;
import org.postgresql.geometric.PGpoint;
import org.postgresql.geometric.PGpolygon;

import java.sql.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Hashtable;

abstract public class Db_vendor_types {
  protected Db_vendor_types() {}
  static protected HashSet<Integer> valid_srid =
          new HashSet<>(Arrays.asList(4326, 900913));

  public static class GisType {
    public String subtype;
    public String type;
    public int srid;
  };

  public abstract boolean isAutoCommitDisabledRequired();
  public abstract GisType find_gis_type(
          Connection conn, ResultSetMetaData metadata, int column_number)
          throws SQLException;
  public abstract String get_wkt(ResultSet rs, int column_number, String gis_type_name)
          throws SQLException;
  public static com.mapd.utility.db_vendors.Db_vendor_types Db_vendor_factory(
          String connection_str) {
    if (connection_str.toLowerCase().contains("postgres"))
      return new com.mapd.utility.db_vendors.PostGis_types();
    else if (connection_str.toLowerCase().contains("omnisci"))
      return new com.mapd.utility.db_vendors.HeavyDBGeo_types();
    return new com.mapd.utility.db_vendors.Other_types();
  }
  public static String gis_type_to_str(GisType type) {
    StringBuffer column_sql_definition = new StringBuffer();
    if (!type.subtype.isEmpty()) {
      column_sql_definition.append(type.subtype + "(");
    }
    column_sql_definition.append(type.type);
    if (type.srid != 0) {
      column_sql_definition.append("," + type.srid);
    }
    if (!type.subtype.isEmpty()) {
      column_sql_definition.append(")");
    }
    return column_sql_definition.toString();
  }
}

class Other_types extends com.mapd.utility.db_vendors.Db_vendor_types {
  protected Other_types() {}
  public boolean isAutoCommitDisabledRequired() {
    return false;
  }
  public GisType find_gis_type(
          Connection conn, ResultSetMetaData metadata, int column_number)
          throws SQLException {
    throw new SQLException("GEO types not supported");
  }
  public String get_wkt(ResultSet rs, int column_number, String gis_type_name)
          throws SQLException {
    throw new SQLException("GEO types not supported");
  }
}

class HeavyDBGeo_types extends com.mapd.utility.db_vendors.Db_vendor_types {
  protected HeavyDBGeo_types() {}

  public boolean isAutoCommitDisabledRequired() {
    return false;
  }
  private static final HashSet<String> geo_types =
          new HashSet<>(Arrays.asList("point", "linestring", "polygon", "multipolygon"));

  // values from SqlTypes.h
  // there seems to be no other way to access those, but IDs are expected NOT to change
  static private Hashtable<Integer, String> subtypes = new Hashtable() {
    {
      put(23, "GEOMETRY");
      put(24, "GEOGRAPHY");
    }
  };

  public String get_wkt(ResultSet rs, int column_number, String gis_type_name)
          throws SQLException {
    return rs.getString(column_number);
  }

  public GisType find_gis_type(
          Connection conn, ResultSetMetaData metadata, int column_number)
          throws SQLException {
    String column_name = metadata.getColumnName(column_number);
    String column_type_name = metadata.getColumnTypeName(column_number);
    if (!geo_types.contains(column_type_name.toLowerCase()))
      throw new SQLException(
              "type not supported: " + column_type_name + " for column " + column_name);
    int srid = metadata.getScale(column_number);
    if (!valid_srid.contains(srid) && srid != 0)
      throw new SQLException(
              "srid is not supported: " + srid + " for column " + column_name);
    int subtype = metadata.getPrecision(column_number);
    if (!subtypes.containsKey(subtype))
      throw new SQLException(
              "Subtype is not supported: " + subtype + " for column " + column_name);
    GisType result = new GisType();
    result.srid = srid;
    result.subtype = subtypes.get(subtype);
    result.type = column_type_name.toUpperCase();
    return result;
  }
}

class PostGis_types extends com.mapd.utility.db_vendors.Db_vendor_types {
  protected PostGis_types() {}
  public boolean isAutoCommitDisabledRequired() {
    return true;
  }

  // Map postgis geom types to HeavyAI geom types
  static private Hashtable<String, String> extra_types = new Hashtable() {
    {
      put("point", "POINT");
      put("lseg", "linestring");
      put("linestring", "linestring");
      put("polygon", "polygon");
      put("multipolygon", "multipolygon");
    }
  };
  private String wkt_point(PGpoint point) {
    return new String("" + point.x + " " + point.y);
  }

  public String get_wkt(ResultSet rs, int column_number, String gis_type_name)
          throws SQLException {
    if (gis_type_name.equalsIgnoreCase("geometry")
            || gis_type_name.equalsIgnoreCase("geography")) {
      Object gis_object = rs.getObject(column_number);
      PGgeometry pGeometry = (PGgeometry) gis_object;
      if (pGeometry == null) throw new SQLException("unknown type");
      // Try and trim the front SRID=nnnn; value from the string  returned from the db.
      // If there isn't a SRID=nnnn; string (marked by the ';') then simply
      // return the whole string
      int semi_colon_indx = pGeometry.getValue().indexOf(';');
      if (-1 != semi_colon_indx && semi_colon_indx < pGeometry.getValue().length()) {
        return pGeometry.getValue().substring(semi_colon_indx + 1);
      }
      return pGeometry.getValue();
    }
    StringBuffer WKT_string = new StringBuffer();
    if (gis_type_name.equalsIgnoreCase("point")) {
      PGpoint point = (PGpoint) rs.getObject(column_number);
      WKT_string.append(extra_types.get(gis_type_name) + "(" + wkt_point(point) + ")");

    } else if (gis_type_name.equalsIgnoreCase("polygon")) {
      PGpolygon polygon = (PGpolygon) rs.getObject(column_number);
      WKT_string.append(extra_types.get(gis_type_name) + "((");
      for (PGpoint p : polygon.points) {
        WKT_string.append(wkt_point(p) + ",");
      }
      WKT_string.replace(WKT_string.length() - 1, WKT_string.length(), "))");
    } else if (gis_type_name.equalsIgnoreCase("lseg")) {
      PGlseg lseg = (PGlseg) rs.getObject(column_number);
      WKT_string.append(extra_types.get(gis_type_name) + "(");
      for (PGpoint p : lseg.point) {
        WKT_string.append(wkt_point(p) + ",");
      }
      WKT_string.replace(WKT_string.length() - 1, WKT_string.length(), ")");
    }
    return WKT_string.toString();
  }

  public GisType find_gis_type(
          Connection conn, ResultSetMetaData metadata, int column_number)
          throws SQLException {
    String column_name = metadata.getColumnName(column_number);
    String table_name = metadata.getTableName(column_number);
    String column_type_name = metadata.getColumnTypeName(column_number);
    if (column_type_name.equalsIgnoreCase("geography"))
      return find_type_detail(
              conn, "geography_columns", "f_geography_column", column_name, table_name);
    else if (column_type_name.equalsIgnoreCase("geometry"))
      return find_type_detail(
              conn, "geometry_columns", "f_geometry_column", column_name, table_name);
    if (!extra_types.containsKey(column_type_name))
      throw new SQLException(
              "type not supported: " + column_type_name + " for column " + column_name);
    GisType result = new GisType();
    result.type = extra_types.get(column_type_name);
    return result;
  }

  private GisType find_type_detail(Connection conn,
          String ref_table_name,
          String ref_column_name,
          String column_name,
          String table_name) throws SQLException {
    Statement detail_st = conn.createStatement();
    // Select for a specific column name from the ref table.
    String select = "select type, srid from " + ref_table_name + " where "
            + ref_column_name + " = '" + column_name + "'";
    if (table_name.length() > 0) {
      select += " and f_table_name"
              + " = '" + table_name + "'";
    }
    ResultSet rs = detail_st.executeQuery(select);
    GisType result = new GisType();
    result.subtype = "GEOMETRY";
    result.type = null;
    result.srid = 0;
    // The select statment above, can return multiple values qualified by schema/table
    // If an outdated driver is used, the utility jar is positioned befor the driver in
    // the classpath or for whatever reason the table_name isn't returned by the driver
    while (rs.next()) {
      String type = rs.getString(1);
      int srid = rs.getInt(2);
      if (result.type == null) {
        result.type = type;
        result.srid = srid;
      } else {
        if (!result.type.equalsIgnoreCase(type) || srid != result.srid) {
          throw new SQLException("multiple column definitions [" + result.type + ":"
                  + type + "] found for column_name [" + column_name + "].\n"
                  + "You can try to switch the jar positions in the classpath "
                  + "or use a more recent postgreSQL driver.");
        }
      }
    }
    if (!extra_types.containsKey(result.type.toLowerCase()))
      throw new SQLException("type not supported");
    result.type = extra_types.get(result.type.toLowerCase()).toUpperCase();
    return result;
  }
}

package com.mapd.utility.db_vendors;

import org.postgis.Geometry;
import org.postgis.PGgeometry;
import org.postgresql.geometric.PGpoint;
import org.postgresql.geometric.PGpolygon;
import org.postgresql.geometric.PGlseg;
import java.sql.*;
import java.util.HashSet;
import java.util.Hashtable;

abstract public class Db_vendor_types {
  protected Db_vendor_types() {}
  static protected HashSet<Integer> valid_srid = new HashSet() {
    {
      add(4326);
      add(900913);
    }
  };

  public abstract String find_gis_type(
          Connection conn, String source_column_name, String source_column_type_name)
          throws SQLException;
  public abstract String get_wkt(ResultSet rs, int column_number, String gis_type_name)
          throws SQLException;
  public static com.mapd.utility.db_vendors.Db_vendor_types Db_vendor_factory(
          String connection_str) {
    if (connection_str.toLowerCase().contains("postgres"))
      return new com.mapd.utility.db_vendors.PostGis_types();
    return null;
  }
}

class PostGis_types extends com.mapd.utility.db_vendors.Db_vendor_types {
  protected PostGis_types() {}

  // Map postgis geom types to OmniSci geom types
  static private Hashtable<String, String> extra_types = new Hashtable() {
    {
      put("point", "POINT");
      put("lseg", "linestring");
      put("linestring", "linestring");
      put("polygon", "polygon");
      put("multipolygon", "multipolyon");
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

  public String find_gis_type(
          Connection conn, String column_name, String column_type_name)
          throws SQLException {
    if (column_type_name.equalsIgnoreCase("geography"))
      return find_type_detail(
              conn, "geography_columns", "f_geography_column", column_name);
    else if (column_type_name.equalsIgnoreCase("geometry"))
      return find_type_detail(conn, "geometry_columns", "f_geometry_column", column_name);
    if (!extra_types.containsKey(column_type_name))
      throw new SQLException("type not supported");
    return extra_types.get(column_type_name);
  }

  private String find_type_detail(Connection conn,
          String ref_table_name,
          String ref_column_name,
          String column_name) throws SQLException {
    String omnisci_type = null;
    Statement detail_st = conn.createStatement();
    // Select for a specific column name from the ref table.
    String select = "select type, srid from " + ref_table_name + "  where "
            + ref_column_name + " = '" + column_name + "'";
    ResultSet rs = detail_st.executeQuery(select);
    String ps_column_type = null;
    int ps_srid = 0;
    // The select statment above, can return multiple values qualified by schema/table.
    // Unfortunately at this stage only the original postgres column name is known.  If
    // get mulitple returns with the same column name, but different types we will not be
    // able to separate which specific column is which type.  This loop checks for this
    // condition and thows when detected.
    while (rs.next()) {
      String type = rs.getString(1);
      int srid = rs.getInt(2);
      // If multiple rows are returned with different geo types for a single coulmn name
      // then throw.
      if (ps_column_type != null
              && (ps_column_type.equalsIgnoreCase(type) || srid != ps_srid)) {
        throw new SQLException("multiple column definitions [" + ps_column_type + ":"
                + type + "] found for column_name [" + column_name + "]");
      }
      ps_column_type = type;
      ps_srid = srid;
    }
    if (!extra_types.containsKey(ps_column_type.toLowerCase()))
      throw new SQLException("type not supported");
    omnisci_type = extra_types.get(ps_column_type.toLowerCase());
    if (ps_srid != 0) {
      if (!valid_srid.contains(new Integer(ps_srid)))
        throw new SQLException("type not supported");
      omnisci_type = omnisci_type + "," + ps_srid;
    }
    return new String("geometry(" + omnisci_type + ")");
  }
}

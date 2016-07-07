/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.jdbc;

import com.mapd.thrift.server.TColumn;
import com.mapd.thrift.server.TColumnData;
import com.mapd.thrift.server.TColumnType;
import com.mapd.thrift.server.TDBInfo;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TEncodingType;
import com.mapd.thrift.server.TQueryResult;
import com.mapd.thrift.server.TRowSet;
import com.mapd.thrift.server.TTypeInfo;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.RowIdLifetime;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author michael
 */
class MapDDatabaseMetaData implements DatabaseMetaData {

  final static Logger logger = LoggerFactory.getLogger(MapDDatabaseMetaData.class);

  MapDConnection con = null;

  public MapDDatabaseMetaData(MapDConnection connection) {
    this.con = connection;
  }

  @Override
  public boolean allProceduresAreCallable() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean allTablesAreSelectable() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public String getURL() throws SQLException { //logger.debug("Entered");
    return con.url;
  }

  @Override
  public String getUserName() throws SQLException { //logger.debug("Entered");
    return con.user;
  }

  @Override
  public boolean isReadOnly() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean nullsAreSortedHigh() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean nullsAreSortedLow() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean nullsAreSortedAtStart() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean nullsAreSortedAtEnd() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public String getDatabaseProductName() throws SQLException { //logger.debug("Entered");
    return "MapD GPU DB";
  }

  @Override
  public String getDatabaseProductVersion() throws SQLException { //logger.debug("Entered");
    try {
      return con.client.get_version();
    } catch (TException ex) {
      throw new SQLException("Failed to get DB version " + ex.toString());
    }
  }

  @Override
  public String getDriverName() throws SQLException { //logger.debug("Entered");
    return "MapD Basic JDBC Driver";
  }

  @Override
  public String getDriverVersion() throws SQLException { //logger.debug("Entered");
    return "0.1";
  }

  @Override
  public int getDriverMajorVersion() {
    return 0;
  }

  @Override
  public int getDriverMinorVersion() {
    return 1;
  }

  @Override
  public boolean usesLocalFiles() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean usesLocalFilePerTable() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsMixedCaseIdentifiers() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean storesUpperCaseIdentifiers() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean storesLowerCaseIdentifiers() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean storesMixedCaseIdentifiers() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsMixedCaseQuotedIdentifiers() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean storesUpperCaseQuotedIdentifiers() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean storesLowerCaseQuotedIdentifiers() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean storesMixedCaseQuotedIdentifiers() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public String getIdentifierQuoteString() throws SQLException { //logger.debug("Entered");
    return " ";
  }

  @Override
  public String getSQLKeywords() throws SQLException { //logger.debug("Entered");
    return "";
  }

  @Override
  public String getNumericFunctions() throws SQLException { //logger.debug("Entered");
    return "";
  }

  @Override
  public String getStringFunctions() throws SQLException { //logger.debug("Entered");
    return "";
  }

  @Override
  public String getSystemFunctions() throws SQLException { //logger.debug("Entered");
    return "";
  }

  @Override
  public String getTimeDateFunctions() throws SQLException { //logger.debug("Entered");
    //return "NOW,CURDATE,SECOND,HOUR,YEAR,EXTRACT,QUARTER,WEEK,MONTH,DATETRUNC";
    return "SECOND, MINUTE, WEEK, HOUR, MONTH, QUARTER, DATETRUNC, NOW, CURDATE, EXTRACT, DATENAME, DATEPART, DATEDIFF, DATETIME";
  }

  @Override
  public String getSearchStringEscape() throws SQLException { //logger.debug("Entered");
    return "\\";
  }

  @Override
  public String getExtraNameCharacters() throws SQLException { //logger.debug("Entered");
    return "";
  }

  @Override
  public boolean supportsAlterTableWithAddColumn() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsAlterTableWithDropColumn() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsColumnAliasing() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean nullPlusNonNullIsNull() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsConvert() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsConvert(int fromType, int toType) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsTableCorrelationNames() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsDifferentTableCorrelationNames() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsExpressionsInOrderBy() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsOrderByUnrelated() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsGroupBy() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsGroupByUnrelated() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsGroupByBeyondSelect() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsLikeEscapeClause() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsMultipleResultSets() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsMultipleTransactions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsNonNullableColumns() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsMinimumSQLGrammar() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsCoreSQLGrammar() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsExtendedSQLGrammar() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsANSI92EntryLevelSQL() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsANSI92IntermediateSQL() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsANSI92FullSQL() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public boolean supportsIntegrityEnhancementFacility() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsOuterJoins() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsFullOuterJoins() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsLimitedOuterJoins() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public String getSchemaTerm() throws SQLException { //logger.debug("Entered");
    return "Database";
  }

  @Override
  public String getProcedureTerm() throws SQLException { //logger.debug("Entered");
    return "N/A";
  }

  @Override
  public String getCatalogTerm() throws SQLException { //logger.debug("Entered");
    return "N/A";
  }

  @Override
  public boolean isCatalogAtStart() throws SQLException { //logger.debug("Entered");
    return true;
  }

  @Override
  public String getCatalogSeparator() throws SQLException { //logger.debug("Entered");
    return ".";
  }

  @Override
  public boolean supportsSchemasInDataManipulation() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSchemasInProcedureCalls() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSchemasInTableDefinitions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSchemasInIndexDefinitions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSchemasInPrivilegeDefinitions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsCatalogsInDataManipulation() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsCatalogsInProcedureCalls() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsCatalogsInTableDefinitions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsCatalogsInIndexDefinitions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsCatalogsInPrivilegeDefinitions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsPositionedDelete() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsPositionedUpdate() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSelectForUpdate() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsStoredProcedures() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSubqueriesInComparisons() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSubqueriesInExists() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSubqueriesInIns() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsSubqueriesInQuantifieds() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsCorrelatedSubqueries() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsUnion() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsUnionAll() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsOpenCursorsAcrossCommit() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsOpenCursorsAcrossRollback() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsOpenStatementsAcrossCommit() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsOpenStatementsAcrossRollback() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public int getMaxBinaryLiteralLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxCharLiteralLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxColumnNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxColumnsInGroupBy() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxColumnsInIndex() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxColumnsInOrderBy() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxColumnsInSelect() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxColumnsInTable() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxConnections() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxCursorNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxIndexLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxSchemaNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxProcedureNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxCatalogNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxRowSize() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public boolean doesMaxRowSizeIncludeBlobs() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public int getMaxStatementLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxStatements() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxTableNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxTablesInSelect() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getMaxUserNameLength() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getDefaultTransactionIsolation() throws SQLException { //logger.debug("Entered");
    return Connection.TRANSACTION_NONE;
  }

  @Override
  public boolean supportsTransactions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsTransactionIsolationLevel(int level) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsDataDefinitionAndDataManipulationTransactions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsDataManipulationTransactionsOnly() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean dataDefinitionCausesTransactionCommit() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean dataDefinitionIgnoredInTransactions() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public ResultSet getProcedures(String catalog, String schemaPattern, String procedureNamePattern) throws SQLException { //logger.debug("Entered");
   throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getProcedureColumns(String catalog, String schemaPattern, String procedureNamePattern, String columnNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  /*
Retrieves a description of the tables available in the given catalog. Only table descriptions matching the catalog, schema, table name and type criteria are returned. They are ordered by TABLE_TYPE, TABLE_CAT, TABLE_SCHEM and TABLE_NAME.
Each table description has the following columns:

TABLE_CAT String => table catalog (may be null)
TABLE_SCHEM String => table schema (may be null)
TABLE_NAME String => table name
TABLE_TYPE String => table type. Typical types are "TABLE", "VIEW", "SYSTEM TABLE", "GLOBAL TEMPORARY", "LOCAL TEMPORARY", "ALIAS", "SYNONYM".
REMARKS String => explanatory comment on the table
TYPE_CAT String => the types catalog (may be null)
TYPE_SCHEM String => the types schema (may be null)
TYPE_NAME String => type name (may be null)
SELF_REFERENCING_COL_NAME String => name of the designated "identifier" column of a typed table (may be null)
REF_GENERATION String => specifies how values in SELF_REFERENCING_COL_NAME are created. Values are "SYSTEM", "USER", "DERIVED". (may be null)
Note: Some databases may not return information for all tables.

Parameters:
catalog - a catalog name; must match the catalog name as it is stored in the database; "" retrieves those without a catalog; null means that the catalog name should not be used to narrow the search
schemaPattern - a schema name pattern; must match the schema name as it is stored in the database; "" retrieves those without a schema; null means that the schema name should not be used to narrow the search
tableNamePattern - a table name pattern; must match the table name as it is stored in the database
types - a list of table types, which must be from the list of table types returned from getTableTypes(),to include; null returns all types
Returns:
ResultSet - each row is a table description
Throws:
SQLException - if a database access error occurs
   */
  @Override
  public ResultSet getTables(String catalog, String schemaPattern, String tableNamePattern, String[] types) throws SQLException { //logger.debug("Entered");

    List<String> tables;
    try {
      tables = con.client.get_tables(con.session);
    } catch (TException ex) {
      throw new SQLException("get_tables failed " + ex.toString());
    }

    TColumnType columns[] = {
      new TColumnType("TABLE_CAT",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TABLE_SCHEM",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TABLE_NAME",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TABLE_TYPE",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("REMARKS",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TYPE_CAT",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TYPE_SCHEM",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TYPE_NAME",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("SELF_REFERENCING_COL_NAME",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("REF_GENERATION",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false))
    };

    Map<String, ArrayList<String>> dataMap = new HashMap(columns.length);
    Map<String, ArrayList<Boolean>> nullMap = new HashMap(columns.length);

    // create component to contain the meta data for the rows
    // and create  a container to store the data and the nul indicators
    List<TColumnType> rowDesc = new ArrayList(columns.length);
    for (TColumnType col : columns) {
      rowDesc.add(col);
      dataMap.put(col.col_name, new ArrayList());
      nullMap.put(col.col_name, new ArrayList());
    }

    // Now add some actual details for table name
    for (String x : tables) {
      dataMap.get("TABLE_NAME").add(x);
      nullMap.get("TABLE_NAME").add(false);

      nullMap.get("TABLE_SCHEM").add(true) ;

      nullMap.get("TABLE_CAT").add(true);

      dataMap.get("TABLE_TYPE").add("TABLE");
      nullMap.get("TABLE_TYPE").add(false);

      nullMap.get("REMARKS").add(true);

      nullMap.get("TYPE_CAT").add(true);

      nullMap.get("TYPE_SCHEM").add(true);

      nullMap.get("TYPE_NAME").add(true);

      nullMap.get("SELF_REFERENCING_COL_NAME").add(true);

      nullMap.get("REF_GENERATION").add(true);

    }

    List<TColumn> columnsList = new ArrayList(columns.length);

    for (TColumnType col : columns) {
      TColumn schemaCol = createTColumnData(dataMap.get(col.col_name), nullMap.get(col.col_name));
      columnsList.add(schemaCol);
    }

    // create a rowset for the result
    TRowSet rowSet = new TRowSet(rowDesc, null, columnsList, true);

    TQueryResult result = new TQueryResult(rowSet, 0, 0, null);

    MapDResultSet tab = new MapDResultSet(result, "GetTables");
    return tab;

  }

  // need to add type to this currently only does str type
  private TColumn createTColumnData(Object data, List<Boolean> nullsList){
    TColumnData colData = new TColumnData();
    colData.setStr_col((List<String>)data);

    TColumn col = new TColumn(colData, nullsList);
    return col;
  }

  @Override
  public ResultSet getSchemas() throws SQLException { //logger.debug("Entered");

    List<TDBInfo> databases = null;

    try {
      databases = con.client.get_databases();
    } catch (TException ex) {
      throw new SQLException("get_database failed " + ex.toString());
    }

    // process info from databses into the resultset, then place in regular return from MapD
    TColumnType columns[] = {
      new TColumnType("TABLE_SCHEM",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TABLE_CATALOG",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false))
    };
    // create component to contain the meta data for the rows
    List<TColumnType> rowDesc = new ArrayList();
    for (TColumnType col : columns) {
      rowDesc.add(col);
    }

    // Now add some actual details for schema name
    List<String> schemaList = new ArrayList();
    List<Boolean> nullList = new ArrayList();
    List<Boolean> catalogNullList = new ArrayList();

    for (TDBInfo x : databases) {
      schemaList.add(x.db_name);
      nullList.add(false);
      catalogNullList.add(true);
    }

    TColumnData colData = new TColumnData();
    colData.setStr_col(schemaList);

    TColumn schemaCol = new TColumn(colData, nullList);
    TColumn catalogCol = new TColumn(null, catalogNullList);

    List<TColumn> columnsList = new ArrayList();
    columnsList.add(schemaCol);
    columnsList.add(catalogCol);

    // create a rowset for the result
    TRowSet rowSet = new TRowSet(rowDesc, null, columnsList, true);

    TQueryResult result = new TQueryResult(rowSet, 0, 0, null);

    MapDResultSet schemas = new MapDResultSet(result, "getSchemas");
    return schemas;
  }

  @Override
  public ResultSet getCatalogs() throws SQLException { //logger.debug("Entered");
    return getSchemas();
  }

  @Override
  public ResultSet getTableTypes() throws SQLException { //logger.debug("Entered");

    TColumnType columns[] = {
      new TColumnType("TABLE_TYPE",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false))
    };

   Map<String, MapDData> dataMap = new HashMap(columns.length);

    // create component to contain the meta data for the rows
    // and create  a container to store the data and the nul indicators
    List<TColumnType> rowDesc = new ArrayList(columns.length);
    for (TColumnType col : columns) {
      rowDesc.add(col);
      dataMap.put(col.col_name, new MapDData(col.col_type.type));
    }

    // Now add some actual details for table name
    dataMap.get("TABLE_TYPE").add("TABLE");

    List<TColumn> columnsList = new ArrayList(columns.length);

    for (TColumnType col : columns) {
      TColumn schemaCol = dataMap.get(col.col_name).getTColumn();
      columnsList.add(schemaCol);
    }

    // create a rowset for the result
    TRowSet rowSet = new TRowSet(rowDesc, null, columnsList, true);

    TQueryResult result = new TQueryResult(rowSet, 0, 0, null);

    MapDResultSet tab = new MapDResultSet(result, "getTableTypes");

    // logger.info("Dump result "+ result.toString());

    return tab;
  }

  /*
  Retrieves a description of table columns available in the specified catalog.
Only column descriptions matching the catalog, schema, table and column name criteria are returned. They are ordered by TABLE_CAT,TABLE_SCHEM, TABLE_NAME, and ORDINAL_POSITION.

Each column description has the following columns:

TABLE_CAT String => table catalog (may be null)
TABLE_SCHEM String => table schema (may be null)
TABLE_NAME String => table name
COLUMN_NAME String => column name
DATA_TYPE int => SQL type from java.sql.Types
TYPE_NAME String => Data source dependent type name, for a UDT the type name is fully qualified
COLUMN_SIZE int => column size.
BUFFER_LENGTH is not used.
DECIMAL_DIGITS int => the number of fractional digits. Null is returned for data types where DECIMAL_DIGITS is not applicable.
NUM_PREC_RADIX int => Radix (typically either 10 or 2)
NULLABLE int => is NULL allowed.
columnNoNulls - might not allow NULL values
columnNullable - definitely allows NULL values
columnNullableUnknown - nullability unknown
REMARKS String => comment describing column (may be null)
COLUMN_DEF String => default value for the column, which should be interpreted as a string when the value is enclosed in single quotes (may be null)
SQL_DATA_TYPE int => unused
SQL_DATETIME_SUB int => unused
CHAR_OCTET_LENGTH int => for char types the maximum number of bytes in the column
ORDINAL_POSITION int => index of column in table (starting at 1)
IS_NULLABLE String => ISO rules are used to determine the nullability for a column.
YES --- if the column can include NULLs
NO --- if the column cannot include NULLs
empty string --- if the nullability for the column is unknown
SCOPE_CATALOG String => catalog of table that is the scope of a reference attribute (null if DATA_TYPE isn't REF)
SCOPE_SCHEMA String => schema of table that is the scope of a reference attribute (null if the DATA_TYPE isn't REF)
SCOPE_TABLE String => table name that this the scope of a reference attribute (null if the DATA_TYPE isn't REF)
SOURCE_DATA_TYPE short => source type of a distinct type or user-generated Ref type, SQL type from java.sql.Types (null if DATA_TYPE isn't DISTINCT or user-generated REF)
IS_AUTOINCREMENT String => Indicates whether this column is auto incremented
YES --- if the column is auto incremented
NO --- if the column is not auto incremented
empty string --- if it cannot be determined whether the column is auto incremented
IS_GENERATEDCOLUMN String => Indicates whether this is a generated column
YES --- if this a generated column
NO --- if this not a generated column
empty string --- if it cannot be determined whether this is a generated column
The COLUMN_SIZE column specifies the column size for the given column. For numeric data, this is the maximum precision. For character data, this is the length in characters. For datetime datatypes, this is the length in characters of the String representation (assuming the maximum allowed precision of the fractional seconds component). For binary data, this is the length in bytes. For the ROWID datatype, this is the length in bytes. Null is returned for data types where the column size is not applicable.

Parameters:
catalog - a catalog name; must match the catalog name as it is stored in the database; "" retrieves those without a catalog; null means that the catalog name should not be used to narrow the search
schemaPattern - a schema name pattern; must match the schema name as it is stored in the database; "" retrieves those without a schema; null means that the schema name should not be used to narrow the search
tableNamePattern - a table name pattern; must match the table name as it is stored in the database
columnNamePattern - a column name pattern; must match the column name as it is stored in the database
Returns:
ResultSet - each row is a column description
Throws:
SQLException - if a database access error occurs
  */
  @Override
  public ResultSet getColumns(String catalog, String schemaPattern, String tableNamePattern, String columnNamePattern) throws SQLException { //logger.debug("Entered");
    //logger.info("TablePattern "+ tableNamePattern+ " columnNamePattern "+ columnNamePattern);
    String modifiedTablePattern = tableNamePattern.replaceAll("%", ".*");
    String modifiedColumnPattern = columnNamePattern.replaceAll("%", ".*");
    //logger.info("TablePattern "+ tableNamePattern+ " modifiedColumnPattern "+ modifiedColumnPattern);

    // declare the columns in the result set
    TColumnType columns[] = {
      new TColumnType("TABLE_CAT",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TABLE_SCHEM",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("TABLE_NAME",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("COLUMN_NAME",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("DATA_TYPE",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("TYPE_NAME",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("COLUMN_SIZE",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("BUFFER_LENGTH",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("DECIMAL_DIGITS",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("NUM_PREC_RADIX",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("NULLABLE",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("REMARKS",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("COLUMN_DEF",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("SQL_DATA_TYPE",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("SQL_DATETIME_SUB",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("CHAR_OCTET_LENGTH",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("ORDINAL_POSITION",
        new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("IS_NULLABLE",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("SCOPE_CATALOG",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("SCOPE_SCHEMA",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("SCOPE_TABLE",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("SOURCE_DATA_TYPE",
        new TTypeInfo(TDatumType.SMALLINT, TEncodingType.NONE, false, false)),
      new TColumnType("IS_AUTOINCREMENT",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("IS_GENERATEDCOLUMN",
        new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false))
    };

    Map<String, MapDData> dataMap = new HashMap(columns.length);

    // create component to contain the meta data for the rows
    // and create  a container to store the data and the nul indicators
    List<TColumnType> rowDesc = new ArrayList(columns.length);
    for (TColumnType col : columns) {
      rowDesc.add(col);
      dataMap.put(col.col_name, new MapDData(col.col_type.type));
    }

    // Now add some actual details for table name
    List<String> tables;
    try {
      tables = con.client.get_tables(con.session);

      for (String tableName : tables) {
        // check if the table matches the input pattern
        if (tableNamePattern == null || tableNamePattern.equals(tableName)) {

          // grab meta data for table
          Map<String, TColumnType> tableDescriptor = con.client.get_table_descriptor(con.session, tableName);

          int ordinal = 0;
          // iterate through the columns
          for (Map.Entry<String, TColumnType> entry : tableDescriptor.entrySet()) {
            TColumnType value = entry.getValue();

            ordinal++;
            if (columnNamePattern == null || entry.getKey().matches(modifiedColumnPattern)) {
              dataMap.get("TABLE_CAT").setNull(true);
              dataMap.get("TABLE_SCHEM").setNull(true);
              dataMap.get("TABLE_NAME").add(tableName);
              dataMap.get("COLUMN_NAME").add(entry.getKey());
              dataMap.get("DATA_TYPE").add(MapDType.toJava(value.col_type.type));
              dataMap.get("TYPE_NAME").add((value.col_type.type.name() + (value.col_type.is_array ? "[]" : "")));
              dataMap.get("COLUMN_SIZE").add(100);
              dataMap.get("BUFFER_LENGTH").setNull(true);
              dataMap.get("DECIMAL_DIGITS").add(10);
              dataMap.get("NUM_PREC_RADIX").add(10);
              dataMap.get("NULLABLE").add(value.col_type.nullable ? DatabaseMetaData.columnNullable:
                      DatabaseMetaData.columnNoNulls);
              dataMap.get("REMARKS").add("Awsome Column");
              dataMap.get("COLUMN_DEF").setNull(true);
              dataMap.get("SQL_DATA_TYPE").add(0);
              dataMap.get("SQL_DATETIME_SUB").setNull(true);
              dataMap.get("CHAR_OCTET_LENGTH").add(0);
              dataMap.get("ORDINAL_POSITION").add(ordinal);
              dataMap.get("IS_NULLABLE").add(value.col_type.nullable ? "YES" : "NO");
              dataMap.get("SCOPE_CATALOG").setNull(true);
              dataMap.get("SCOPE_SCHEMA").setNull(true);
              dataMap.get("SCOPE_TABLE").setNull(true);
              dataMap.get("SOURCE_DATA_TYPE").add(MapDType.toJava(value.col_type.type));
              dataMap.get("IS_AUTOINCREMENT").add("NO");
              dataMap.get("IS_GENERATEDCOLUMN").add("NO");
            }
          }
        }
      }
    } catch (TException ex) {
      throw new SQLException("get_tables failed " + ex.toString());
    }

    List<TColumn> columnsList = new ArrayList(columns.length);

    for (TColumnType col : columns) {
      TColumn schemaCol = dataMap.get(col.col_name).getTColumn();
      //logger.info("Tcolumn is "+ schemaCol.toString());
      columnsList.add(schemaCol);
    }

    // create a rowset for the result
    TRowSet rowSet = new TRowSet(rowDesc, null, columnsList, true);

    TQueryResult result = new TQueryResult(rowSet, 0, 0, null);

    MapDResultSet cols = new MapDResultSet(result, "getColumns");
    return cols;
  }

  @Override
  public ResultSet getColumnPrivileges(String catalog, String schema, String table, String columnNamePattern) throws SQLException { //logger.debug("Entered");
   throw new UnsupportedOperationException("Not supported yet.");
  }

  public ResultSet getEmptyResultSet(){
    return new MapDResultSet();
  }

  @Override
  public ResultSet getTablePrivileges(String catalog, String schemaPattern, String tableNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getBestRowIdentifier(String catalog, String schema, String table, int scope, boolean nullable) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getVersionColumns(String catalog, String schema, String table) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getPrimaryKeys(String catalog, String schema, String table) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getImportedKeys(String catalog, String schema, String table) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getExportedKeys(String catalog, String schema, String table) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getCrossReference(String parentCatalog, String parentSchema, String parentTable, String foreignCatalog, String foreignSchema, String foreignTable) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  /*
  Retrieves a description of all the data types supported by this database. They are ordered by DATA_TYPE and then by how closely the data type maps to the corresponding JDBC SQL type.
If the database supports SQL distinct types, then getTypeInfo() will return a single row with a TYPE_NAME of DISTINCT and a DATA_TYPE of Types.DISTINCT. If the database supports SQL structured types, then getTypeInfo() will return a single row with a TYPE_NAME of STRUCT and a DATA_TYPE of Types.STRUCT.

If SQL distinct or structured types are supported, then information on the individual types may be obtained from the getUDTs() method.

Each type description has the following columns:

TYPE_NAME String => Type name
DATA_TYPE int => SQL data type from java.sql.Types
PRECISION int => maximum precision
LITERAL_PREFIX String => prefix used to quote a literal (may be null)
LITERAL_SUFFIX String => suffix used to quote a literal (may be null)
CREATE_PARAMS String => parameters used in creating the type (may be null)
NULLABLE short => can you use NULL for this type.
typeNoNulls - does not allow NULL values
typeNullable - allows NULL values
typeNullableUnknown - nullability unknown
CASE_SENSITIVE boolean=> is it case sensitive.
SEARCHABLE short => can you use "WHERE" based on this type:
typePredNone - No support
typePredChar - Only supported with WHERE .. LIKE
typePredBasic - Supported except for WHERE .. LIKE
typeSearchable - Supported for all WHERE ..
UNSIGNED_ATTRIBUTE boolean => is it unsigned.
FIXED_PREC_SCALE boolean => can it be a money value.
AUTO_INCREMENT boolean => can it be used for an auto-increment value.
LOCAL_TYPE_NAME String => localized version of type name (may be null)
MINIMUM_SCALE short => minimum scale supported
MAXIMUM_SCALE short => maximum scale supported
SQL_DATA_TYPE int => unused
SQL_DATETIME_SUB int => unused
NUM_PREC_RADIX int => usually 2 or 10
The PRECISION column represents the maximum column size that the server supports for the given datatype. For numeric data, this is the maximum precision. For character data, this is the length in characters. For datetime datatypes, this is the length in characters of the String representation (assuming the maximum allowed precision of the fractional seconds component). For binary data, this is the length in bytes. For the ROWID datatype, this is the length in bytes. Null is returned for data types where the column size is not applicable.

Returns:
a ResultSet object in which each row is an SQL type description
Throws:
SQLException - if a database access error occurs
  */
  @Override
  public ResultSet getTypeInfo() throws SQLException { //logger.debug("Entered");

    // declare the columns in the result set
    TColumnType columns[] = {
      new TColumnType("TYPE_NAME",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("DATA_TYPE",
      new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("PRECISION",
      new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("LITERAL_PREFIX",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("LITERAL_SUFFIX",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("CREATE_PARAMS",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("NULLABLE",
      new TTypeInfo(TDatumType.SMALLINT, TEncodingType.NONE, false, false)),
      new TColumnType("CASE_SENSITIVE",
      new TTypeInfo(TDatumType.BOOL, TEncodingType.NONE, false, false)),
      new TColumnType("SEARCHABLE",
      new TTypeInfo(TDatumType.SMALLINT, TEncodingType.NONE, false, false)),
      new TColumnType("UNSIGNED_ATTRIBUTE",
      new TTypeInfo(TDatumType.BOOL, TEncodingType.NONE, false, false)),
      new TColumnType("FIXED_PREC_SCALE",
      new TTypeInfo(TDatumType.BOOL, TEncodingType.NONE, false, false)),
      new TColumnType("AUTO_INCREMENT",
      new TTypeInfo(TDatumType.BOOL, TEncodingType.NONE, false, false)),
      new TColumnType("LOCAL_TYPE_NAME",
      new TTypeInfo(TDatumType.STR, TEncodingType.NONE, false, false)),
      new TColumnType("MINIMUM_SCALE",
      new TTypeInfo(TDatumType.SMALLINT, TEncodingType.NONE, false, false)),
      new TColumnType("MAXIMUM_SCALE",
      new TTypeInfo(TDatumType.SMALLINT, TEncodingType.NONE, false, false)),
      new TColumnType("SQL_DATA_TYPE",
      new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("SQL_DATETIME_SUB",
      new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false)),
      new TColumnType("NUM_PREC_RADIX",
      new TTypeInfo(TDatumType.INT, TEncodingType.NONE, false, false))
    };

    Map<String, MapDData> dataMap = new HashMap(columns.length);

    // create component to contain the meta data for the rows
    // and create  a container to store the data and the nul indicators
    List<TColumnType> rowDesc = new ArrayList(columns.length);
    for (TColumnType col : columns) {
      rowDesc.add(col);
      dataMap.put(col.col_name, new MapDData(col.col_type.type));
    }
//TODO this is currently a work in progress need to add actual details here
    // Now add some actual details for table name
    dataMap.get("TYPE_NAME").setNull(true);// String => Type name
    dataMap.get("DATA_TYPE").setNull(true);// int => SQL data type from java.sql.Types
    dataMap.get("PRECISION").setNull(true);// int => maximum precision
    dataMap.get("LITERAL_PREFIX").setNull(true);//.setNull(true);// String => prefix used to quote a literal (may be null)
    dataMap.get("LITERAL_SUFFIX").setNull(true);//.setNull(true);// String => suffix used to quote a literal (may be null)
    dataMap.get("CREATE_PARAMS").setNull(true);// String => parameters used in creating the type (may be null)
    dataMap.get("NULLABLE").setNull(true);// short => can you use NULL for this type.
//typeNoNulls - does not allow NULL values
//typeNullable - allows NULL values
//typeNullableUnknown - nullability unknown
    dataMap.get("CASE_SENSITIVE").setNull(true);// boolean=> is it case sensitive.
    dataMap.get("SEARCHABLE").setNull(true);// short => can you use "WHERE" based on this type:
//typePredNone - No support
//typePredChar - Only supported with WHERE .. LIKE
//typePredBasic - Supported except for WHERE .. LIKE
//typeSearchable - Supported for all WHERE ..
    dataMap.get("UNSIGNED_ATTRIBUTE").setNull(true);// boolean => is it unsigned.
    dataMap.get("FIXED_PREC_SCALE").setNull(true);// boolean => can it be a money value.
    dataMap.get("AUTO_INCREMENT").setNull(false);// boolean => can it be used for an auto-increment value.
    dataMap.get("LOCAL_TYPE_NAME").setNull(true);// String => localized version of type name (may be null)
    dataMap.get("MINIMUM_SCALE").setNull(true);// short => minimum scale supported
    dataMap.get("MAXIMUM_SCALE").setNull(true);// short => maximum scale supported
    dataMap.get("SQL_DATA_TYPE").setNull(true);// int => unused
    dataMap.get("SQL_DATETIME_SUB").setNull(true);// int => unused
    dataMap.get("NUM_PREC_RADIX").setNull(true);//

    List<TColumn> columnsList = new ArrayList(columns.length);

    for (TColumnType col : columns) {
      TColumn schemaCol = dataMap.get(col.col_name).getTColumn();
      // logger.info("Tcolumn is "+ schemaCol.toString());
      columnsList.add(schemaCol);
    }

    // create a rowset for the result
    TRowSet rowSet = new TRowSet(rowDesc, null, columnsList, true);

    TQueryResult result = new TQueryResult(rowSet, 0, 0, null);

    MapDResultSet cols = new MapDResultSet(result, "getTypeInfo");
    return cols;

  }

  @Override
  public ResultSet getIndexInfo(String catalog, String schema, String table, boolean unique, boolean approximate) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean supportsResultSetType(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsResultSetConcurrency(int type, int concurrency) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean ownUpdatesAreVisible(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean ownDeletesAreVisible(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean ownInsertsAreVisible(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean othersUpdatesAreVisible(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean othersDeletesAreVisible(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean othersInsertsAreVisible(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean updatesAreDetected(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean deletesAreDetected(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean insertsAreDetected(int type) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsBatchUpdates() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public ResultSet getUDTs(String catalog, String schemaPattern, String typeNamePattern,
          int[] types) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public Connection getConnection() throws SQLException { //logger.debug("Entered");
    return con;
  }

  @Override
  public boolean supportsSavepoints() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsNamedParameters() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsMultipleOpenResults() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsGetGeneratedKeys() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public ResultSet getSuperTypes(String catalog, String schemaPattern, String typeNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getSuperTables(String catalog, String schemaPattern, String tableNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getAttributes(String catalog, String schemaPattern, String typeNamePattern, String attributeNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean supportsResultSetHoldability(int holdability) throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public int getResultSetHoldability() throws SQLException { //logger.debug("Entered");
    return ResultSet.CLOSE_CURSORS_AT_COMMIT;
  }

  @Override
  public int getDatabaseMajorVersion() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getDatabaseMinorVersion() throws SQLException { //logger.debug("Entered");
    return 1;
  }

  @Override
  public int getJDBCMajorVersion() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public int getJDBCMinorVersion() throws SQLException { //logger.debug("Entered");
    return 1;
  }

  @Override
  public int getSQLStateType() throws SQLException { //logger.debug("Entered");
    return 0;
  }

  @Override
  public boolean locatorsUpdateCopy() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean supportsStatementPooling() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public RowIdLifetime getRowIdLifetime() throws SQLException { //logger.debug("Entered");
    return RowIdLifetime.ROWID_VALID_OTHER;
  }

  @Override
  public ResultSet getSchemas(String catalog, String schemaPattern) throws SQLException { //logger.debug("Entered");
    return getSchemas();
  }

  @Override
  public boolean supportsStoredFunctionsUsingCallSyntax() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public boolean autoCommitFailureClosesAllResultSets() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public ResultSet getClientInfoProperties() throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getFunctions(String catalog, String schemaPattern, String functionNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getFunctionColumns(String catalog, String schemaPattern, String functionNamePattern, String columnNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public ResultSet getPseudoColumns(String catalog, String schemaPattern, String tableNamePattern, String columnNamePattern) throws SQLException { //logger.debug("Entered");
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean generatedKeyAlwaysReturned() throws SQLException { //logger.debug("Entered");
    return false;
  }

  @Override
  public <T> T unwrap(Class<T> iface) throws SQLException { //logger.debug("Entered");
    return null;
  }

  @Override
  public boolean isWrapperFor(Class<?> iface) throws SQLException { //logger.debug("Entered");
    return false;
  }

}

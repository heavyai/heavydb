<#--
 Copyright 2022 HEAVY.AI, Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

/*
 * Create a new foreign table using the one of the following syntaxes:
 *
 * CREATE FOREIGN TABLE [ IF NOT EXISTS ] <table_name> (
 *   { <column_name> <data_type> [NOT NULL] [ENCODING <encoding_spec>]
 *   [ WITH ( <option> = <value> [, ... ] ) ] }
 *   [, ... ]
 * )
 * SERVER <server_name>
 * [ WITH ( <option> = <value> [, ... ] ) ]
 *
 * ..or..
 *
 * CREATE FOREIGN TABLE [ IF NOT EXISTS ] <table_name>
 * SCHEMA <schema_name>
 * SERVER <server_name>
 * [ WITH ( <option> = <value> [, ... ] ) ]
 */
SqlCreateForeignTable SqlCreateForeignTable(Span s, boolean replace) :
{
    boolean ifNotExists = false;
    SqlIdentifier tableName = null;
    SqlIdentifier serverName = null;
    HeavySqlSanitizedString schemaName = null;
    SqlNode schemaLit = null;
    List<HeavySqlColumn> columns = null;
    HeavySqlOptionsMap columnOptions = null;
}
{
    <FOREIGN> <TABLE>
    ifNotExists = IfNotExistsOpt()
    tableName = CompoundIdentifier()
    (
      <SCHEMA>
      schemaLit = Literal()
      { schemaName = new HeavySqlSanitizedString(schemaLit); }
    |
      columns = Columns()
    )
    <SERVER>
    serverName = CompoundIdentifier()
    [ columnOptions = WithOptions() ]
    {
	return new SqlCreateForeignTable(s.end(this), ifNotExists, tableName,
					 serverName,
					 schemaName,
                                         columns, columnOptions);
    }
}

/*
 * Parse a squence of one or more comma-separated HeavySqlColumns.
 *
 * (
 *   { <column_name> <data_type> [NOT NULL] [ENCODING <encoding_spec>]
 *   [ WITH ( <option> = <value> [, ... ] ) ] }
 *   [, ... ]
 * )
 */
List<HeavySqlColumn> Columns() :
{
    List<HeavySqlColumn> columns = new ArrayList<HeavySqlColumn>();
    HeavySqlColumn column = null;
}
{
    <LPAREN>
    column = Column()
    { columns.add(column); }
    (
        <COMMA>
	column = Column()
	{ columns.add(column); }
    )*
    <RPAREN>
    { return columns; }
}

/*
 * Parse a single HeavySqlColumn.
 *
 * <column_name> <data_type> [NOT NULL] [ENCODING <encoding_spec>]
 * [ WITH ( <option> = <value> [, ... ] ) ]
 */
HeavySqlColumn Column() :
{
    SqlIdentifier columnName;
    HeavySqlDataType dataType;
    HeavySqlEncoding encodingType = null;
    HeavySqlOptionsMap optionsMap = null;
}
{
    columnName = CompoundIdentifier()
    dataType = HeavyDataType()
    [ optionsMap = WithOptions() ]
    { return new HeavySqlColumn(columnName, dataType, encodingType, optionsMap); }
}

/*
 * Parse array syntax given a type.
 *
 * \[ [ <size> ] \]
 *
 */
HeavySqlArray HeavyArray(SqlDataTypeSpec type) :
{
    Integer size = null;
}
{
    <LBRACKET>
    (
        size = UnsignedIntLiteral()
        <RBRACKET>
    |
        <RBRACKET>
    )
    { return new HeavySqlArray(type.toString(), size); }
}

HeavySqlTypeNameSpec HeavyDBGeoDataType():
{
    String typeName = null;
    Integer coordinate = null;
}
{
    (
        typeName = HeavyDBGeoDataTypeName()
    |
        <GEOMETRY>
        <LPAREN>
        typeName = HeavyDBGeoDataTypeName()
        [
            <COMMA>
            coordinate = IntLiteral()
        ]
        <RPAREN>
    )
    {
        return new HeavySqlTypeNameSpec(typeName, coordinate, SqlTypeName.GEOMETRY, getPos());
    }
}

String HeavyDBGeoDataTypeName() :
{
    Token type = null;
}
{
    (
        type = <POINT>
    |
        type = <LINESTRING>
    |
        type = <POLYGON>
    |
        type = <MULTIPOLYGON>
    )
    {
        return type.toString();
    }
}

HeavySqlTypeNameSpec HeavyDBText() : {}
{
    <TEXT> { return new HeavySqlTypeNameSpec("TEXT", SqlTypeName.VARCHAR, getPos()); }
}

/*
 * Parse an HeavyDB datatype with optional not-null attribute.
 *
 * <data_type>[ \[ [<size>] \] ] [NOT NULL] [ENCODING <encoding_spec>]
 */
HeavySqlDataType HeavyDataType() :
{
    SqlDataTypeSpec type;
    HeavySqlArray array = null;
    Integer coordinate = null;
    HeavySqlEncoding encoding = null;
    boolean notNull = false;
}
{
    type = DataType()
    [
        array = HeavyArray(type)
    ]
    [ <NOT> <NULL> { notNull = true; } ]
    [ encoding = Encoding() ]
    { return new HeavySqlDataType(type, notNull, array, encoding); }
}

/*
 * Parse a datatype with optional not-null attribute.
 *
 * <data_type> [NOT NULL]
 */
SqlDataTypeSpec NullableDataType() :
{
    SqlDataTypeSpec type;
}
{
    type = DataType()
    [ <NOT> <NULL> { return type.withNullable(false); } ]
    { return type.withNullable(true); }
}

/*
 * Parse an HeavySql encoding.
 *
 * ENCODING <encoding_spec>
 */
HeavySqlEncoding Encoding() :
{
    Token type = null;
    Integer size = null;
}
{
    <ENCODING>
    (
        type = <NONE>
    |
        ( type = <FIXED> | type = <DAYS> )
        <LPAREN> size = IntLiteral() <RPAREN>
    |
        ( type = <DICT> | type = <COMPRESSED> )
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    )
    { return new HeavySqlEncoding(type.toString(), size); }
}

// TODO: Foreign table and server use different object types
// for processing options. Update to use the same methods/object types

/*
 * Parse one or more options following the SET keyword.
 *
 * SET ( <option> = <value> [, ... ] )
 */
HeavySqlOptionsMap SetOptions() :
{
    HeavySqlOptionsMap optionMap = new HeavySqlOptionsMap();
}
{
  <SET>
  optionMap = TableOptions()
  { return optionMap; }
}

/*
 * Parse one or more HeavySqlOptions following the WITH keyword.
 *
 * WITH ( <option> = <value> [, ... ] )
 */
HeavySqlOptionsMap WithOptions() :
{
    HeavySqlOptionsMap optionMap = new HeavySqlOptionsMap();
}
{
  <WITH>
  optionMap = TableOptions()
  { return optionMap; }
}

HeavySqlOptionsMap TableOptions() :
{
    HeavySqlOptionsMap optionMap = new HeavySqlOptionsMap();
    HeavySqlOptionPair optionPair = null;
}
{
  <LPAREN>
  optionPair = TableOption()
  { optionMap.put(optionPair.getKey(), optionPair.getValue().toString()); }
  (
    <COMMA>
    optionPair = TableOption()
    { optionMap.put(optionPair.getKey(), optionPair.getValue().toString()); }
  )*
  <RPAREN>
  { return optionMap; }
}

/*
 * Parse an HeavySqlOption.
 *
 * <option> = <value>
 */
HeavySqlOptionPair TableOption() :
{
    final SqlIdentifier option;
    final String optionString;
    final SqlNode value;
}
{
    (
      // Special rule required to handle "escape" option, since ESCAPE is a keyword
      <ESCAPE>
      {
        optionString = "escape";
      }
    |
      option = CompoundIdentifier()
      {
        optionString = option.toString();
      }
    )
    <EQ>
    value = Literal()
    { return new HeavySqlOptionPair(optionString,
                                   new HeavySqlSanitizedString(value)); }
}

/*
 * Drop a new foreign table using the following syntax:
 *
 * DROP FOREIGN TABLE [ IF EXISTS ] <table_name>
 */
SqlDrop SqlDropForeignTable(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier tableName;
}
{
    <FOREIGN> <TABLE>
    ifExists = IfExistsOpt()
    tableName = CompoundIdentifier()
    {
        return new SqlDropForeignTable(s.end(this), ifExists, tableName.toString());
    }
}

/*
 * Refresh the cache for a foreign table using the following syntax.
 * If the evict flag is set, then we perform a cache flush instead (no reload).
 *
 * REFRESH FOREIGN TABLES <table_name> [, ... ]* [ WITH (EVICT = 'true') ]
 */
SqlDdl SqlRefreshForeignTables(Span s) : {
    List<String> tableNames = new ArrayList<String>();
    SqlIdentifier tableName = null;
    HeavySqlOptionsMap optionsMap = null;
}
{
  <REFRESH> <FOREIGN> <TABLES>
  tableName = CompoundIdentifier()
  { tableNames.add(tableName.toString()); }
  (
   <COMMA>
   tableName = CompoundIdentifier()
   { tableNames.add(tableName.toString()); }
  )*
  [ optionsMap = WithOptions() ]
  { return new SqlRefreshForeignTables(s.end(this), tableNames, optionsMap); }
}

/**
 * ALTER FOREIGN TABLE DDL syntax variants:
 *
 * ALTER FOREIGN TABLE <table> [ SET (<option> = <value> [, ... ] ) ]
 * ALTER FOREIGN TABLE <table> RENAME TO <new_table>
 * ALTER FOREIGN TABLE <table> RENAME COLUMN <old_column> TO <new_column>
 */
SqlDdl SqlAlterForeignTable(Span s) :
{
    SqlAlterForeignTable.Builder sqlAlterForeignTableBuilder =
        new SqlAlterForeignTable.Builder();
    SqlIdentifier tableName;
    SqlIdentifier newTableName;
    SqlIdentifier oldColumnName;
    SqlIdentifier newColumnName;
    HeavySqlOptionsMap optionsMap = null;
}
{
    <ALTER> <FOREIGN> <TABLE>
    tableName=CompoundIdentifier()
    { sqlAlterForeignTableBuilder.setTableName(tableName.toString()); }
    (
        <RENAME>
        (
            <TO>
            newTableName=CompoundIdentifier()
            { sqlAlterForeignTableBuilder.alterTableName(newTableName.toString()); }
        |
            <COLUMN>
            oldColumnName=CompoundIdentifier()
            <TO>
            newColumnName=CompoundIdentifier()
            {
                sqlAlterForeignTableBuilder.alterColumnName(oldColumnName.toString(),
                                                            newColumnName.toString());
            }
        )
    |
        optionsMap = SetOptions()
        { sqlAlterForeignTableBuilder.alterOptions(optionsMap); }
    )
    {
        sqlAlterForeignTableBuilder.setPos(s.end(this));
        return sqlAlterForeignTableBuilder.build();
    }
}

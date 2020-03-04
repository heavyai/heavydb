<#--
 Copyright 2020 OmniSci, Inc.

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
    OmniSqlSanitizedString schemaName = null;
    SqlNode schemaLit = null;
    List<OmniSqlColumn> columns = null;
    OmniSqlOptionsMap columnOptions = null;
}
{
    <FOREIGN> <TABLE>
    ifNotExists = IfNotExistsOpt()
    tableName = CompoundIdentifier()
    (
      <SCHEMA>
      schemaLit = Literal()
      { schemaName = new OmniSqlSanitizedString(schemaLit); }
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
 * Parse a squence of one or more comma-separated OmniSqlColumns.
 *
 * (
 *   { <column_name> <data_type> [NOT NULL] [ENCODING <encoding_spec>]
 *   [ WITH ( <option> = <value> [, ... ] ) ] }
 *   [, ... ]
 * )
 */
List<OmniSqlColumn> Columns() :
{
    List<OmniSqlColumn> columns = new ArrayList<OmniSqlColumn>();
    OmniSqlColumn column = null;
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
 * Parse a single OmniSciColumn.
 *
 * <column_name> <data_type> [NOT NULL] [ENCODING <encoding_spec>]
 * [ WITH ( <option> = <value> [, ... ] ) ]
 */
OmniSqlColumn Column() :
{
    SqlIdentifier columnName;
    OmniSqlDataType dataType;
    OmniSqlEncoding encodingType = null;
    OmniSqlOptionsMap optionsMap = null;
}
{
    columnName = CompoundIdentifier()
    dataType = OmniDataType()
    [ optionsMap = WithOptions() ]
    { return new OmniSqlColumn(columnName, dataType, encodingType, optionsMap); }
}

/*
 * Parse array syntax given a type.
 *
 * \[ [ <size> ] \]
 *
 */
OmniSqlArray OmniArray(SqlDataTypeSpec type) :
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
    { return new OmniSqlArray(type, size); }
}

OmniSqlTypeNameSpec OmniPoint() : {}
{
    <POINT> { return new OmniSqlTypeNameSpec("POINT", SqlTypeName.GEOMETRY, getPos()); }
}

OmniSqlTypeNameSpec OmniLineString() : {}
{
    <LINESTRING> { return new OmniSqlTypeNameSpec("LINESTRING", SqlTypeName.GEOMETRY, getPos()); }
}

OmniSqlTypeNameSpec OmniPolygon() : {}
{
    <POLYGON> { return new OmniSqlTypeNameSpec("POLYGON", SqlTypeName.GEOMETRY, getPos()); }
}

OmniSqlTypeNameSpec OmniMultiPolygon() : {}
{
    <MULTIPOLYGON> { return new OmniSqlTypeNameSpec("MULTIPOLYGON", SqlTypeName.GEOMETRY, getPos()); }
}

OmniSqlTypeNameSpec OmniText() : {}
{
    <TEXT> { return new OmniSqlTypeNameSpec("TEXT", SqlTypeName.VARCHAR, getPos()); }
}
/*
OmniSqlTypeNameSpec OmniDecimal() : {}
{
    <DECIMAL> { return new OmniSqlTypeNameSpec("DECIMAL", SqlTypeName.DECIMAL, getPos()); }
}
*/

OmniSqlTypeNameSpec OmniEpoch() : {}
{
    <EPOCH> { return new OmniSqlTypeNameSpec("EPOCH", SqlTypeName.DATE, getPos()); }
}

/*
 * Parse an Omnisci datatype with optional not-null attribute.
 *
 * <data_type>[ \[ [<size>] \] ] [NOT NULL] [ENCODING <encoding_spec>]
 */
OmniSqlDataType OmniDataType() :
{
    SqlDataTypeSpec type;
    OmniSqlArray array = null;
    Integer precision = null;
    Integer scale = null;
    Integer coordinate = null;
    OmniSqlEncoding encoding = null;
    boolean notNull = false;
}
{
    type = DataType()
    [(
        array = OmniArray(type)
    |
        /* This line causes an ambiguous parser warning when
           DECIMAL is a custom type, but we have no way of
           extracting the precision information that we want if
           we use default DECIMAL parsing.
        */
        // TODO:  This is only for GEO.
        <LPAREN>
        type = DataType()
        [ <COMMA> coordinate = UnsignedIntLiteral() ]
        <RPAREN>
        // TODO:  This is for custom decimal extension.
        /*|
            precision = UnsignedIntLiteral()
            [
                <COMMA>
                scale = UnsignedIntLiteral()
                ]
            <RPAREN>
            )*/
    )]
    [ <NOT> <NULL> { notNull = true; } ]
    [ encoding = Encoding() ]
    { return new OmniSqlDataType(type, notNull, array, precision, scale, coordinate, encoding); }
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
 * Parse an OmniSql encoding.
 *
 * ENCODING <encoding_spec>
 */
OmniSqlEncoding Encoding() :
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
    { return new OmniSqlEncoding(type.toString(), size); }
}

/*
 * Parse one or more OmniSqlOptions following the WITH keyword.
 *
 * WITH ( <option> = <value> [, ... ] )
 */
OmniSqlOptionsMap WithOptions() :
{
    OmniSqlOptionsMap optionMap = new OmniSqlOptionsMap();
    OmniSqlOptionPair optionPair = null;
}
{
  <WITH> <LPAREN>
  optionPair = WithOption()
  { optionMap.put(optionPair.getKey(), optionPair.getValue().toString()); }
  (
    <COMMA>
    optionPair = WithOption()
    { optionMap.put(optionPair.getKey(), optionPair.getValue().toString()); }
  )*
  <RPAREN>
  { return optionMap; }
}

/*
 * Parse an OmniSqlOption.
 *
 * <option> = <value>
 */
OmniSqlOptionPair WithOption() :
{
    final SqlIdentifier withOption;
    final SqlNode withValue;
}
{
    withOption = CompoundIdentifier()
    <EQ>
    withValue = Literal()
    { return new OmniSqlOptionPair(withOption.toString(),
                                   new OmniSqlSanitizedString(withValue)); }
}

/*
 * Drop a new foreign table using the following syntax:
 *
 * DROP FOREIGN TABLE [ IF EXISTS ] <table_name>
 */
SqlDrop SqlDropForeignTable(Span s, boolean replace) :
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

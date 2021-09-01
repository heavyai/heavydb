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




SqlNodeList DropColumnNodeList() :
{
    SqlIdentifier id;
    final List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    [<COLUMN>]
    id = SimpleIdentifier() {list.add(id);}
    (
        <COMMA> [<DROP>][<COLUMN>] id = SimpleIdentifier() {
            list.add(id);
        }
    )*
    { return new SqlNodeList(list, SqlParserPos.ZERO); }
}

// TableElementList without parens
SqlNodeList NoParenTableElementList() :
{
    final Span s;
    final List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    { s = span(); }
    TableElement(list)
    (
        <COMMA> TableElement(list)
    )*
    {
        return new SqlNodeList(list, s.end(this));
    }
}

SqlNodeList TableElementList() :
{
    final Span s;
    final List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    <LPAREN> { s = span(); }
    TableElement(list)
    (
        <COMMA> TableElement(list)
    )*
    <RPAREN> {
        return new SqlNodeList(list, s.end(this));
    }
}

SqlNodeList idList() :
{
    final Span s;
    SqlIdentifier id;
    final List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    { s = span(); }
    id = CompoundIdentifier() { list.add(id); }
    (
        <COMMA> 
        id = CompoundIdentifier() { list.add(id); }
    )*
    {
        return new SqlNodeList(list, s.end(this));
    }
}

// Parse an optional data type encoding, default is NONE.
Pair<OmniSciEncoding, Integer> OmniSciEncodingOpt() :
{
    OmniSciEncoding encoding;
    Integer size = 0;
}
{
    <ENCODING>
    (
        <NONE> { encoding = OmniSciEncoding.NONE; }
    |
        <FIXED> { encoding = OmniSciEncoding.FIXED; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    |
        <DAYS> { encoding = OmniSciEncoding.DAYS; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    | 
        <DICT> { encoding = OmniSciEncoding.DICT; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    | 
        <COMPRESSED> { encoding = OmniSciEncoding.COMPRESSED; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    )
    { return new Pair(encoding, size); }
}

// Parse sql type name that allow arrays
SqlTypeNameSpec OmniSciArrayTypeName(Span s) :
{
    final SqlTypeName sqlTypeName;
    boolean isText = false;
    Integer size = -1;
}
{
    (
        <BOOLEAN> { sqlTypeName = SqlTypeName.BOOLEAN; }
    |
        ( <INTEGER> | <INT> ) { sqlTypeName = SqlTypeName.INTEGER; }
    |
        <TINYINT> { sqlTypeName = SqlTypeName.TINYINT; }
    |
        <SMALLINT> { sqlTypeName = SqlTypeName.SMALLINT; }
    |
        <BIGINT> { sqlTypeName = SqlTypeName.BIGINT; }
    |
        <DOUBLE> { sqlTypeName = SqlTypeName.DOUBLE; }
    |
        <FLOAT> { sqlTypeName = SqlTypeName.FLOAT; }
    |
        <TEXT> { 
            isText = true;
            sqlTypeName = SqlTypeName.VARCHAR; // TODO: consider overloading / adding TEXT as base type
        }
    |
        <DATE> { sqlTypeName = SqlTypeName.DATE; }
    |
        <TIME> { sqlTypeName = SqlTypeName.TIME; }
    )
     <LBRACKET>
    (
        size = UnsignedIntLiteral()
        <RBRACKET>
    |
        <RBRACKET>
    )
    {
        return new OmniSciTypeNameSpec(sqlTypeName, isText, size, s.end(this));
    }
}

// Parse DECIMAL that allows arrays
SqlTypeNameSpec OmniSciDecimalArrayTypeName(Span s) :
{
    final SqlTypeName sqlTypeName;
    Integer size = -1;
    Integer precision = -1;
    Integer scale = -1;
}
{
    (<DECIMAL> | <DEC> | <NUMERIC>) { sqlTypeName = SqlTypeName.DECIMAL; }
    (
        <LPAREN>
        precision = UnsignedIntLiteral()
        [
            <COMMA>
            scale = UnsignedIntLiteral()
        ]
        <RPAREN>
    )
     <LBRACKET>
    (
        size = UnsignedIntLiteral()
        <RBRACKET>
    |
        <RBRACKET>
    )
    {
        return new OmniSciTypeNameSpec(sqlTypeName, false, size, precision, scale, s.end(this));
    }
}

// Parse sql TIMESTAMP that allow arrays 
SqlTypeNameSpec OmniSciTimestampArrayTypeName(Span s) :
{
    final SqlTypeName sqlTypeName;
    Integer size = -1;
    Integer precision = -1;
    final Integer scale = -1;
}
{
    <TIMESTAMP> { sqlTypeName = SqlTypeName.TIMESTAMP; }
    [
        <LPAREN>
        precision = UnsignedIntLiteral()
        <RPAREN>
    ]
     <LBRACKET>
    (
        size = UnsignedIntLiteral()
        <RBRACKET>
    |
        <RBRACKET>
    )
    {
        return new OmniSciTypeNameSpec(sqlTypeName, false, size, precision, scale, s.end(this));
    }
}

// Parse sql TEXT type 
SqlTypeNameSpec OmniSciTextTypeName(Span s) :
{
    final SqlTypeName sqlTypeName;
    boolean isText = false;
}
{
    
    <TEXT> { 
        isText = true;
        sqlTypeName = SqlTypeName.VARCHAR; // TODO: consider overloading / adding TEXT as base type
    }
    {
        return new OmniSciTypeNameSpec(sqlTypeName, isText, s.end(this));
    }
}

OmniSciGeo OmniSciGeoType() :
{
    final OmniSciGeo geoType;
}
{
    (
        <POINT> { geoType = OmniSciGeo.POINT; }
    |   
        <LINESTRING> { geoType = OmniSciGeo.LINESTRING; }
    | 
        <POLYGON> { geoType = OmniSciGeo.POLYGON; }
    |
        <MULTIPOLYGON> { geoType = OmniSciGeo.MULTIPOLYGON; }
    )
    {
        return geoType;
    }
}

// Parse sql type name for geospatial data 
SqlTypeNameSpec OmniSciGeospatialTypeName(Span s) :
{
    final OmniSciGeo geoType;
    boolean isGeography = false;
    Integer coordinateSystem = 0;
    Pair<OmniSciEncoding, Integer> encoding = null;
}
{
    (
        geoType = OmniSciGeoType()
        |
        (
            <GEOMETRY> { 
                isGeography = false;
            }
        |
            <GEOGRAPHY> { 
                isGeography = true;
            }
        )
        <LPAREN> geoType = OmniSciGeoType() [ <COMMA> coordinateSystem = IntLiteral() ] <RPAREN>
        [ encoding = OmniSciEncodingOpt() ]
    )
    {
        return new OmniSciGeoTypeNameSpec(geoType, coordinateSystem, isGeography, encoding, s.end(this));
    }
}

// Some SQL type names need special handling due to the fact that they have
// spaces in them but are not quoted.
SqlTypeNameSpec OmniSciTypeName() :
{
    final SqlTypeNameSpec typeNameSpec;
    final SqlIdentifier typeName;
    final Span s = Span.of();
}
{
    (
<#-- additional types are included here -->
<#-- put custom data types in front of Calcite core data types -->
        LOOKAHEAD(2)
        typeNameSpec = OmniSciArrayTypeName(s)
    |
        LOOKAHEAD(5)
        typeNameSpec = OmniSciTimestampArrayTypeName(s)
    |
        LOOKAHEAD(7)
        typeNameSpec = OmniSciDecimalArrayTypeName(s)
    |
        LOOKAHEAD(2)
        typeNameSpec = OmniSciTextTypeName(s)
    |
        LOOKAHEAD(2)
        typeNameSpec = OmniSciGeospatialTypeName(s)
    |
<#list parser.dataTypeParserMethods as method>
        LOOKAHEAD(2)
        typeNameSpec = ${method}
    |
</#list>
        LOOKAHEAD(2)
        typeNameSpec = SqlTypeName(s)
    |
        typeNameSpec = RowTypeName()
    |
        typeName = CompoundIdentifier() {
            typeNameSpec = new SqlUserDefinedTypeNameSpec(typeName, s.end(this));
        }
    )
    {
        return typeNameSpec;
    }
}


// Type name with optional scale and precision.
OmniSciSqlDataTypeSpec OmniSciDataType() :
{
    SqlTypeNameSpec typeName;
    final Span s;
}
{
    typeName = OmniSciTypeName() {
        s = span();
    }
    (
        typeName = CollectionsTypeName(typeName)
    )*
    {
        return new OmniSciSqlDataTypeSpec(
            typeName,
            s.end(this));
    }
}

void OmniSciShardKeyOpt(List<SqlNode> list) : 
{
    final Span s;
    final SqlIdentifier name;
}
{
    <SHARD> { s = span(); } <KEY> 
    <LPAREN> name = SimpleIdentifier() {
        list.add(SqlDdlNodes.shard(s.end(this), name));
    }
    <RPAREN> 
}

SqlIdentifier OmniSciSharedDictReferences() :
{
    final Span s = Span.of();
    SqlIdentifier name;
    final SqlIdentifier name2;
}
{
    (
        LOOKAHEAD(2)
        name = SimpleIdentifier() 
        <LPAREN> name2 = SimpleIdentifier() <RPAREN> {
            name = name.add(1, name2.getSimple(), s.end(this));
        }
    |
        name = SimpleIdentifier()
    )
    {
        return name;
    }
}

void OmniSciSharedDictOpt(List<SqlNode> list) :
{
    final Span s;
    final SqlIdentifier columnName;
    final SqlIdentifier referencesColumn;
}
{
    <SHARED> { s = span(); } <DICTIONARY>
    <LPAREN> columnName = SimpleIdentifier() <RPAREN>
    <REFERENCES> 
    referencesColumn = OmniSciSharedDictReferences()
    {
        list.add(SqlDdlNodes.sharedDict(s.end(this), columnName, referencesColumn));
    }
}

void TableElement(List<SqlNode> list) :
{
    final SqlIdentifier id;
    final OmniSciSqlDataTypeSpec type;
    final boolean nullable;
    Pair<OmniSciEncoding, Integer> encoding = null;
    final SqlNode defval;
    final SqlNode constraint;
    SqlIdentifier name = null;
    final Span s = Span.of();
    final ColumnStrategy strategy;
}
{
    (
        LOOKAHEAD(3)
        OmniSciShardKeyOpt(list)
    |
        OmniSciSharedDictOpt(list)
    |
        (
            id = SimpleIdentifier()
            (
                type = OmniSciDataType()
                nullable = NullableOptDefaultTrue()
                (
                    <DEFAULT_> defval = Expression(ExprContext.ACCEPT_SUB_QUERY) {
                        strategy = ColumnStrategy.DEFAULT;
                    }
                |
                    {
                        defval = null;
                        strategy = nullable ? ColumnStrategy.NULLABLE
                            : ColumnStrategy.NOT_NULLABLE;
                    }
                )
                [ encoding = OmniSciEncodingOpt() ]
                {
                    list.add(
                        SqlDdlNodes.column(s.add(id).end(this), id,
                            type.withEncoding(encoding).withNullable(nullable),
                            defval, strategy));
                }
            |
                { list.add(id); }
            )
        )
    )
}

/*
 * Parse an option key/value pair. The value is expected to be a non-interval literal. 
 *
 * <option> = <value>
 */
Pair<String, SqlNode> KVOption() :
{
    final SqlIdentifier option;
    final String optionString;
    final SqlNode value;
    final Span s;
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
    { s = span(); }
    (
        <NULL> {  value = SqlLiteral.createCharString("", s.end(this)); }
    |
        value = Literal()
    )
    { return new Pair<String, SqlNode>(optionString, value); }
}

/*
 * Parse one or more key-value pair options
 *
 * ( <option> = <value> [, ... ] )
 */
OmniSciOptionsMap OptionsOpt() :
{
    OmniSciOptionsMap optionMap = new OmniSciOptionsMap();
    Pair<String, SqlNode> optionPair = null;
}
{
  <LPAREN>
  optionPair = KVOption()
  { OmniSciOptionsMap.add(optionMap, optionPair.getKey(), optionPair.getValue()); }
  (
    <COMMA>
    optionPair = KVOption()
    { OmniSciOptionsMap.add(optionMap, optionPair.getKey(), optionPair.getValue()); }
  )*
  <RPAREN>
  { return optionMap; }
}

/*
 * Parse the TEMPORARY keyphrase.
 *
 * [ TEMPORARY ]
 */
boolean TemporaryOpt() :
{
}
{
    <TEMPORARY> { return true; }
|
    { return false; }
}

/*
 * Create a table using the following syntax:
 *
 * CREATE TABLE [ IF NOT EXISTS ] <table_name> AS <select>
 */
SqlCreate SqlCreateTable(Span s, boolean replace) :
{
    boolean temporary = false;
    boolean ifNotExists = false;
    final SqlIdentifier id;
    SqlNodeList tableElementList = null;
    OmniSciOptionsMap withOptions = null;
    SqlNode query = null;
}
{
    [<TEMPORARY> {temporary = true; }]
    <TABLE> 
    ifNotExists = IfNotExistsOpt() 
    id = CompoundIdentifier()
    ( 
        tableElementList = TableElementList()
    |
        <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
    )
    [ <WITH> withOptions = OptionsOpt() ]
    {
        return SqlDdlNodes.createTable(s.end(this), replace, temporary, ifNotExists, id,
            tableElementList, withOptions, query);
    }
}

/**
 * Parses a DROP statement.
 *
 *  This broke away from the default Calcite implementation because it was
 *  necessary to use LOOKAHEAD(2) for the USER commands in order for them 
 *  to parse correctly.
 *
 *   "replace" was never used, but appeared in SqlDrop's original signature
 *
 */
SqlDdl SqlCustomDrop(Span s) :
{
    boolean replace = false;  
    final SqlDdl drop;
}
{
    <DROP>
    (
        LOOKAHEAD(1) drop = SqlDropDB(s)
        |
        LOOKAHEAD(1) drop = SqlDropTable(s)
        |
        LOOKAHEAD(1) drop = SqlDropView(s)
        |
        LOOKAHEAD(1) drop = SqlDropServer(s)
        |
        LOOKAHEAD(1) drop = SqlDropForeignTable(s)
        |
        LOOKAHEAD(2) drop = SqlDropUserMapping(s)
        |
        LOOKAHEAD(2) drop = SqlDropUser(s)
        |
        LOOKAHEAD(2) drop = SqlDropRole(s)
    )
    {
        return drop;
    }
}

/*
 * Drop a table using the following syntax:
 *
 * DROP TABLE [ IF EXISTS ] <table_name>
 */
SqlDdl SqlDropTable(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier tableName;
}
{
    <TABLE>
    ifExists = IfExistsOpt()
    tableName = CompoundIdentifier()
    {
        return new SqlDropTable(s.end(this), ifExists, tableName.toString());
    }
}


/*
 * Rename table(s) using the following syntax:
 *
 * RENAME TABLE <table_name> TO <new_table_name> [, <table_name_n> TO <new_table_name_n>]
 */
SqlRenameTable SqlRenameTable(Span s) :
{   
    SqlIdentifier tableName;
    SqlIdentifier newTableName;
    final List<Pair<String, String>> tableNames = new ArrayList<Pair<String, String>>();
}
{
    <RENAME>
    <TABLE>
    tableName = CompoundIdentifier()
    <TO>
    newTableName = CompoundIdentifier()
    { tableNames.add(new Pair<String, String>(tableName.toString(), newTableName.toString())); }
    (
        <COMMA>
        tableName = CompoundIdentifier()
        <TO>
        newTableName = CompoundIdentifier()
        { tableNames.add(new Pair<String, String>(tableName.toString(), newTableName.toString())); }
    )*
    {
        return new SqlRenameTable(s.end(this), tableNames);
    }
}

/*
 * Alter a table using the following syntax:
 *
 * ALTER TABLE <table_name>
 *
 */
SqlDdl SqlAlterTable(Span s) :
{
    SqlAlterTable.Builder sqlAlterTableBuilder = new SqlAlterTable.Builder();
    SqlIdentifier tableName;
    SqlIdentifier newTableName;
    SqlIdentifier columnName;
    SqlIdentifier newColumnName;
    SqlIdentifier columnType;
    SqlIdentifier encodingSpec;
    boolean notNull = false;
    SqlNodeList columnList = null;
}
{
    <ALTER>
    <TABLE>
    tableName=CompoundIdentifier()
    {
        sqlAlterTableBuilder.setTableName(tableName.toString());
    }
    (
        <RENAME>
        (
            <TO>
            newTableName = CompoundIdentifier()
            {
                sqlAlterTableBuilder.alterTableName(newTableName.toString());
            }
        |
            <COLUMN>
            columnName = CompoundIdentifier()
            <TO>
            newColumnName = CompoundIdentifier()
            {
                sqlAlterTableBuilder.alterColumnName(columnName.toString(), newColumnName.toString());
            }
        )
    |
        <DROP>
        columnList = DropColumnNodeList()
        {
            sqlAlterTableBuilder.dropColumn(columnList);
        }
    |
        <ADD>
        [<COLUMN>]
        (
            columnList = TableElementList()
            |
            columnList = NoParenTableElementList()
        )
        {
            sqlAlterTableBuilder.addColumnList(columnList);
        }
    |
        <SET>
        Option(sqlAlterTableBuilder)
        (
            <COMMA>
            Option(sqlAlterTableBuilder)
        )*
        {
            sqlAlterTableBuilder.alterOptions();
        }
    )
    {
        sqlAlterTableBuilder.setPos(s.end(this));

        // Builder implementation
        return sqlAlterTableBuilder.build();

    }
}

SqlNode WrappedOrderedQueryOrExpr(ExprContext exprContext) :
{
    final SqlNode query;
}
{
        <LPAREN>
        query = OrderedQueryOrExpr(exprContext)
        <RPAREN>
        { return query; }
    |
        query = OrderedQueryOrExpr(exprContext)
        { return query; }
}

/* 
 * Insert into table(s) using the following syntax:
 *
 * INSERT INTO <table_name> [columns] <select>
 */
SqlInsertIntoTable SqlInsertIntoTable(Span s) :
{
    final SqlIdentifier table;
    final SqlNode query;
    SqlNodeList columnList = null;
}
{
    <INSERT>
    <INTO>
    table = CompoundIdentifier()
    (
        LOOKAHEAD(3)
        columnList = ParenthesizedSimpleIdentifierList() 
        query = WrappedOrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
    |
        query = WrappedOrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
    )

    {
        return new SqlInsertIntoTable(s.end(this), table, query, columnList);
    }
}

/*
 * Dump a table using the following syntax:
 *
 * DUMP TABLE <tableName> [WITH options]
 *
 */
SqlDdl SqlDumpTable(Span s) :
{
    final SqlIdentifier tableName;
    OmniSciOptionsMap withOptions = null;
    SqlNode filePath = null;
}
{
    ( <DUMP> | <ARCHIVE> )
    <TABLE>
    tableName = CompoundIdentifier()
    <TO>
    filePath = StringLiteral()
    [ <WITH> withOptions = OptionsOpt() ]
    {
        return new SqlDumpTable(s.end(this), tableName.toString(), filePath.toString(), withOptions);
    }
}

/*
 * Restore a table using the following syntax:
 *
 * RESTORE TABLE <tableName> [WITH options]
 *
 */
SqlDdl SqlRestoreTable(Span s) :
{
    final SqlIdentifier tableName;
    OmniSciOptionsMap withOptions = null;
    SqlNode filePath = null; 
}
{
    <RESTORE>
    <TABLE>
    tableName = CompoundIdentifier()
    <FROM>
    filePath = StringLiteral()
    [ <WITH> withOptions = OptionsOpt() ]
    {
        return new SqlRestoreTable(s.end(this), tableName.toString(), filePath.toString(), withOptions);
    }
}

/*
 * Truncate a table using the following syntax:
 *
 * TRUNCATE TABLE <tableName>
 *
 */
SqlDdl SqlTruncateTable(Span s) :
{
    final SqlIdentifier tableName;
}
{
    <TRUNCATE>
    <TABLE>
    tableName = CompoundIdentifier()
    {       
        return new SqlTruncateTable(s.end(this), tableName.toString());
    }
}

/*
 * Optimize a table using the following syntax:
 *
 * OPTIMIZE TABLE <tableName> [WITH options]
 *
 */
SqlDdl SqlOptimizeTable(Span s) :
{
    final SqlIdentifier tableName;
    OmniSciOptionsMap withOptions = null;   
}
{
    <OPTIMIZE>
    <TABLE>
    tableName = CompoundIdentifier()
    [ <WITH> withOptions = OptionsOpt() ]
    {
        return new SqlOptimizeTable(s.end(this), tableName.toString(), withOptions);
    }
}


/*
 * Create a view using the following syntax:
 *
 * CREATE VIEW [ IF NOT EXISTS ] <view_name> [(columns)] AS <query>
 */
SqlCreate SqlCreateView(Span s, boolean replace) :
{
    final boolean ifNotExists;
    final SqlIdentifier id;
    SqlNodeList columnList = null;
    final SqlNode query;
}
{
    <VIEW> ifNotExists = IfNotExistsOpt() id = CompoundIdentifier()
    [ columnList = ParenthesizedSimpleIdentifierList() ]
    <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY) {
        if (columnList != null && columnList.size() > 0) {
            throw new ParseException("Column list aliases in views are not yet supported.");
        }
        return SqlDdlNodes.createView(s.end(this), replace, ifNotExists, id, columnList,
            query);
    }
}

/*
 * Drop a view using the following syntax:
 *
 * DROP VIEW [ IF EXISTS ] <view_name>
 */
SqlDdl SqlDropView(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier viewName;
}
{
    <VIEW>
    ifExists = IfExistsOpt()
    viewName = CompoundIdentifier()
    {
        return new SqlDropView(s.end(this), ifExists, viewName.toString());
    }
}

/*
 * Create a database using the following syntax:
 *
 * CREATE DATABASE ...
 *
 *  "replace" option required by SqlCreate, but unused
 */
SqlCreate SqlCreateDB(Span s, boolean replace) :
{
    final boolean ifNotExists;
    final SqlIdentifier dbName;
    OmniSciOptionsMap dbOptions = null;
}
{
    <DATABASE> ifNotExists = IfNotExistsOpt() dbName = CompoundIdentifier()
    [ dbOptions = OptionsOpt() ]
    {
        return new SqlCreateDB(s.end(this), ifNotExists, dbName.toString(), dbOptions);
    }
}

/* 
 * Drop a database using the following syntax:
 *
 * DROP DATABASE [ IF EXISTS ] <db_name>
 *
 */
SqlDdl SqlDropDB(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier dbName;
}
{
    <DATABASE>
    ifExists = IfExistsOpt()
    dbName = CompoundIdentifier()
    {
        return new SqlDropDB(s.end(this), ifExists, dbName.toString());
    }
}


/*
 * Rename database using the following syntax:
 *
 * ALTER DATABASE <db_name> RENAME TO <new_db_name>
 */
SqlRenameDB SqlRenameDB(Span s) :
{   
    SqlIdentifier dbName;
    SqlIdentifier newDBName;
}
{
    <ALTER>
    <DATABASE>
    dbName = CompoundIdentifier()
    <RENAME>
    <TO>
    newDBName = CompoundIdentifier()
    {
        return new SqlRenameDB(s.end(this), dbName.toString(), newDBName.toString());
    }
}

/*
 * Create a user using the following syntax:
 *
 * CREATE USER ["]<name>["] (<property> = value,...);
 *
 *  "replace" option required by SqlCreate, but unused
 */
SqlCreate SqlCreateUser(Span s, boolean replace) :
{
    final SqlIdentifier userName;
    OmniSciOptionsMap userOptions = null;
}
{
    <USER> 
    userName = CompoundIdentifier()
    [ userOptions = OptionsOpt() ]
    {
        return new SqlCreateUser(s.end(this), userName.toString(), userOptions);
    }
}


/* 
 * Drop a user using the following syntax:
 *
 * DROP USER <user_name>
 *
 */
SqlDdl SqlDropUser(Span s) :
{
    final SqlIdentifier userName;
}
{
    <USER>
    userName = CompoundIdentifier()
    {
        return new SqlDropUser(s.end(this), userName.toString());
    }
}


/*
 * Alter user using the following syntax:
 *
 * ALTER USER <user_name> RENAME TO <new_user_name>
 * ALTER USER <user_name> dbOptions
 *
 */
SqlDdl SqlAlterUser(Span s) :
{   
    SqlIdentifier userName;
    SqlIdentifier newUserName = null;
    OmniSciOptionsMap userOptions = null;
}
{
    <ALTER>
    <USER>
    userName = CompoundIdentifier()
    (
        <RENAME>
        <TO>
        newUserName = CompoundIdentifier()
    |
        userOptions = OptionsOpt()
    )
    {
        if(userOptions != null){
            return new SqlAlterUser(s.end(this), userName.toString(), userOptions);           
        } else {
            return new SqlRenameUser(s.end(this), userName.toString(), newUserName.toString()); 
        }
    }
}

/*
 * Create a role using the following syntax:
 *
 * CREATE ROLE <role_name>
 *
 */
SqlCreate SqlCreateRole(Span s, boolean replace) :
{
    final SqlIdentifier role;
}
{
    <ROLE> role = CompoundIdentifier()
    {
        return new SqlCreateRole(s.end(this), role);
    }
}

/*
 * Drop a role using the following syntax:
 *
 * DROP ROLE <role_name>
 *
 */
SqlDrop SqlDropRole(Span s) :
{
    final SqlIdentifier role;
}
{
    <ROLE> role = CompoundIdentifier()
    {
        return new SqlDropRole(s.end(this), role.toString());
    }
}

/*
 * Base class for:
 *
 * Grant using the following syntax:
 *
 * GRANT <nodeList> ...
 *
 */
SqlDdl SqlGrant(Span si) :
{
    Span s;
    SqlNodeList nodeList;
    final SqlDdl grant;
}
{
    <GRANT> { s = span(); }
    (
        nodeList = privilegeList()
        grant = SqlGrantPrivilege(s, nodeList)
        |
        nodeList = idList()
        grant = SqlGrantRole(s, nodeList)
    )
    {
        return grant;
    }
}

/*
 * Base class for:
 *
 * Revoke using the following syntax:
 *
 * REVOKE <nodeList> ...
 *
 */
SqlDdl SqlRevoke(Span si) :
{
    Span s;
    SqlNodeList nodeList;
    final SqlDdl revoke;
}
{
    <REVOKE> { s = span(); }
    (
        nodeList = privilegeList()
        revoke = SqlRevokePrivilege(s, nodeList)
        |
        nodeList = idList()
        revoke = SqlRevokeRole(s, nodeList)
    )
    {
        return revoke;
    }
}

/*
 * Grant a role using the following syntax:
 *
 * (GRANT <rolenames>) TO <grantees>
 *
 */
SqlDdl SqlGrantRole(Span s, SqlNodeList roleList) :
{
    SqlNodeList granteeList;
}
{
    <TO>
    granteeList = idList()

    { return new SqlGrantRole(s.end(this), roleList, granteeList); }
}

/*
 * Revoke a role using the following syntax:
 *
 * (REVOKE <rolenames>) FROM <grantees>
 *
 */
SqlDdl SqlRevokeRole(Span s, SqlNodeList roleList) :
{
    SqlNodeList granteeList;
}
{
    <FROM>
    granteeList = idList()

    { return new SqlRevokeRole(s.end(this), roleList, granteeList); }
}

SqlNode Privilege(Span s) :
{
    String type;
}
{
    (
        LOOKAHEAD(2) <SERVER> <USAGE> { type = "SERVER USAGE"; }
    |   LOOKAHEAD(2) <ALTER> <SERVER> { type = "ALTER SERVER"; }
    |   LOOKAHEAD(2) <CREATE> <SERVER> { type = "CREATE SERVER"; }
    |   LOOKAHEAD(2) <CREATE> <TABLE> { type = "CREATE TABLE"; }
    |   LOOKAHEAD(2) <CREATE> <VIEW> { type = "CREATE VIEW"; }
    |   LOOKAHEAD(2) <SELECT> <VIEW> { type = "SELECT VIEW"; }
    |   LOOKAHEAD(2) <DROP> <VIEW> { type = "DROP VIEW"; }
    |   LOOKAHEAD(2) <DROP> <SERVER> { type = "DROP SERVER"; }
    |   LOOKAHEAD(2) <CREATE> <DASHBOARD> { type = "CREATE DASHBOARD"; }
    |   LOOKAHEAD(2) <EDIT> <DASHBOARD> { type = "EDIT DASHBOARD"; }
    |   LOOKAHEAD(2) <VIEW> <DASHBOARD> { type = "VIEW DASHBOARD"; }
    |   LOOKAHEAD(2) <DELETE> <DASHBOARD> { type = "DELETE DASHBOARD"; }
    |   LOOKAHEAD(2) <VIEW> <SQL> <EDITOR> { type = "VIEW SQL EDITOR"; }
    |   LOOKAHEAD(2) <ALL> [<PRIVILEGES>] { type = "ALL"; }
    |   <ACCESS> { type = "ACCESS"; }
    |   <ALTER> { type = "ALTER"; }
    |   <CREATE> { type = "CREATE"; }
    |   <DELETE> { type = "DELETE"; }
    |   <DROP> { type = "DROP"; }
    |   <EDIT> { type = "EDIT"; }
    |   <INSERT> { type = "INSERT"; }
    |   <SELECT> { type = "SELECT"; }
    |   <TRUNCATE> { type = "TRUNCATE"; }
    |   <UPDATE> { type = "UPDATE"; }
    |   <USAGE> { type = "USAGE"; }
    |   <VIEW> { type = "VIEW"; }
    )
    { return SqlLiteral.createCharString(type, s.end(this)); }
}

SqlNodeList privilegeList() :
{
    Span s;
    SqlNode privilege;
    final List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    { s = span(); }
    privilege = Privilege(s) { list.add(privilege); }
    (
        <COMMA> 
        privilege = Privilege(s) { list.add(privilege); }
    )*
    {
        return new SqlNodeList(list, s.end(this));
    }
}

String PrivilegeType() :
{
    String type;
}
{
    (
        <DATABASE> { type = "DATABASE"; }
    |   <TABLE> { type = "TABLE"; }
    |   <DASHBOARD> { type = "DASHBOARD"; }
    |   <VIEW> { type = "VIEW"; }
    |   <SERVER> { type = "SERVER"; }
    )
    { return type; }
}

String PrivilegeTarget() :
{
    final SqlIdentifier target;
    final Integer iTarget;
    final String result;
}
{
    (
        target = CompoundIdentifier() { result = target.toString(); }
    |   
        iTarget = IntLiteral() { result = iTarget.toString(); }
    )
    { return result; }
}

/*
 * Grant privileges using the following syntax:
 *
 * (GRANT privileges) ON privileges_target_type privileges_target TO grantees
 *
 */
SqlDdl SqlGrantPrivilege(Span s, SqlNodeList privileges) :
{
    final String type;
    final String target;
    final SqlNodeList grantees;
}
{
    <ON>
    type = PrivilegeType()
    target = PrivilegeTarget()
    <TO>
    grantees = idList()
    {
        return new SqlGrantPrivilege(s.end(this), privileges, type, target, grantees);
    }
}

/*
 * Revoke privileges using the following syntax:
 *
 * (REVOKE <privileges>) ON <type> <target> FROM <entityList>;
 *
 */
SqlDdl SqlRevokePrivilege(Span s, SqlNodeList privileges) :
{
    final String type;
    final String target;
    final SqlNodeList grantees;
}
{
    <ON>
    type = PrivilegeType()
    target = PrivilegeTarget()
    <FROM>
    grantees = idList()
    {
        return new SqlRevokePrivilege(s.end(this), privileges, type, target, grantees);
    }
}



/*
 * Reassign owned database objects using the following syntax:
 *
 * REASSIGN OWNED BY <old_owner>, <old_owner>, ... TO <new_owner>
 */
SqlDdl SqlReassignOwned(Span s) :
{
    SqlIdentifier userName = null;
    List<String> oldOwners = null;
}
{
    <REASSIGN> <OWNED> <BY>
    userName = CompoundIdentifier()
    {
        oldOwners = new ArrayList<String>();
        oldOwners.add(userName.toString());
    }
    (
        <COMMA>
        userName = CompoundIdentifier()
        {
            oldOwners.add(userName.toString());
        }
    )*
    <TO>
    userName = CompoundIdentifier()
    {
        return new SqlReassignOwned(s.end(this), oldOwners, userName.toString());
    }
}

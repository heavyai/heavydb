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
void AlterTableColumElement(List<SqlNode> list) :
{
    SqlIdentifier id;
}
{
    <ALTER>
    [<COLUMN>]
    id = SimpleIdentifier()
    [<SET> <DATA>]
    <TYPE>
    TableColumn(id,list)
}

SqlNodeList RepeatedAlterColumnList() :
{
    final Span s;
    final List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    { s = span(); }
    AlterTableColumElement(list)
    (
        <COMMA>
        AlterTableColumElement(list)
    )*
    {
        return new SqlNodeList(list, s.end(this));
    }
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
    id = HyphenatedCompoundIdentifier() { list.add(id); }
    (
        <COMMA> 
        id = HyphenatedCompoundIdentifier() { list.add(id); }
    )*
    {
        return new SqlNodeList(list, s.end(this));
    }
}

// Parse an optional data type encoding, default is NONE.
Pair<HeavyDBEncoding, Integer> HeavyDBEncodingOpt() :
{
    HeavyDBEncoding encoding;
    Integer size = 0;
}
{
    <ENCODING>
    (
        <NONE> { encoding = HeavyDBEncoding.NONE; }
    |
        <FIXED> { encoding = HeavyDBEncoding.FIXED; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    |
        <DAYS> { encoding = HeavyDBEncoding.DAYS; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    | 
        <DICT> { encoding = HeavyDBEncoding.DICT; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    | 
        <COMPRESSED> { encoding = HeavyDBEncoding.COMPRESSED; }
        [ <LPAREN> size = IntLiteral() <RPAREN> ]
    )
    { return new Pair(encoding, size); }
}

// Parse sql type name that allow arrays
SqlTypeNameSpec HeavyDBArrayTypeName(Span s) :
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
        return new HeavyDBTypeNameSpec(sqlTypeName, isText, size, s.end(this));
    }
}

// Parse DECIMAL that allows arrays
SqlTypeNameSpec HeavyDBDecimalArrayTypeName(Span s) :
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
        return new HeavyDBTypeNameSpec(sqlTypeName, false, size, precision, scale, s.end(this));
    }
}

// Parse sql TIMESTAMP that allow arrays 
SqlTypeNameSpec HeavyDBTimestampArrayTypeName(Span s) :
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
        return new HeavyDBTypeNameSpec(sqlTypeName, false, size, precision, scale, s.end(this));
    }
}

// Parse sql TEXT type 
SqlTypeNameSpec HeavyDBTextTypeName(Span s) :
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
        return new HeavyDBTypeNameSpec(sqlTypeName, isText, s.end(this));
    }
}

HeavyDBGeo HeavyDBGeoType() :
{
    final HeavyDBGeo geoType;
}
{
    (
        <POINT> { geoType = HeavyDBGeo.POINT; }
    |   
        <MULTIPOINT> { geoType = HeavyDBGeo.MULTIPOINT; }
    |   
        <LINESTRING> { geoType = HeavyDBGeo.LINESTRING; }
    | 
        <MULTILINESTRING> { geoType = HeavyDBGeo.MULTILINESTRING; }
    | 
        <POLYGON> { geoType = HeavyDBGeo.POLYGON; }
    |
        <MULTIPOLYGON> { geoType = HeavyDBGeo.MULTIPOLYGON; }
    )
    {
        return geoType;
    }
}

// Parse sql type name for geospatial data 
SqlTypeNameSpec HeavyDBGeospatialTypeName(Span s) :
{
    final HeavyDBGeo geoType;
    boolean isGeography = false;
    Integer coordinateSystem = 0;
    Pair<HeavyDBEncoding, Integer> encoding = null;
}
{
    (
        geoType = HeavyDBGeoType()
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
        <LPAREN> geoType = HeavyDBGeoType() [ <COMMA> coordinateSystem = IntLiteral() ] <RPAREN>
        [ encoding = HeavyDBEncodingOpt() ]
    )
    {
        return new HeavyDBGeoTypeNameSpec(geoType, coordinateSystem, isGeography, encoding, s.end(this));
    }
}

// Some SQL type names need special handling due to the fact that they have
// spaces in them but are not quoted.
SqlTypeNameSpec HeavyDBTypeName() :
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
        typeNameSpec = HeavyDBArrayTypeName(s)
    |
        LOOKAHEAD(5)
        typeNameSpec = HeavyDBTimestampArrayTypeName(s)
    |
        LOOKAHEAD(7)
        typeNameSpec = HeavyDBDecimalArrayTypeName(s)
    |
        LOOKAHEAD(2)
        typeNameSpec = HeavyDBTextTypeName(s)
    |
        LOOKAHEAD(2)
        typeNameSpec = HeavyDBGeospatialTypeName(s)
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
HeavyDBSqlDataTypeSpec HeavyDBDataType() :
{
    SqlTypeNameSpec typeName;
    final Span s;
}
{
    typeName = HeavyDBTypeName() {
        s = span();
    }
    (
        typeName = CollectionsTypeName(typeName)
    )*
    {
        return new HeavyDBSqlDataTypeSpec(
            typeName,
            s.end(this));
    }
}

void HeavyDBShardKeyOpt(List<SqlNode> list) : 
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

SqlIdentifier HeavyDBSharedDictReferences() :
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

void HeavyDBSharedDictOpt(List<SqlNode> list) :
{
    final Span s;
    final SqlIdentifier columnName;
    final SqlIdentifier referencesColumn;
}
{
    <SHARED> { s = span(); } <DICTIONARY>
    <LPAREN> columnName = SimpleIdentifier() <RPAREN>
    <REFERENCES> 
    referencesColumn = HeavyDBSharedDictReferences()
    {
        list.add(SqlDdlNodes.sharedDict(s.end(this), columnName, referencesColumn));
    }
}

void TableElement(List<SqlNode> list) :
{
    final SqlIdentifier id;
    final HeavyDBSqlDataTypeSpec type;
    final boolean nullable;
    Pair<HeavyDBEncoding, Integer> encoding = null;
    final SqlNode defval;
    final SqlNode constraint;
    SqlIdentifier name = null;
    final Span s = Span.of();
    final ColumnStrategy strategy;
}
{
    (
        LOOKAHEAD(3)
        HeavyDBShardKeyOpt(list)
    |
        HeavyDBSharedDictOpt(list)
    |
        (
            id = SimpleIdentifier()
            TableColumn(id,list)
        )
    )
}

void TableColumn(SqlIdentifier id, List<SqlNode> list) :
{
    final HeavyDBSqlDataTypeSpec type;
    final boolean nullable;
    Pair<HeavyDBEncoding, Integer> encoding = null;
    final SqlNode defval;
    final SqlNode constraint;
    SqlIdentifier name = null;
    final Span s = Span.of();
    final ColumnStrategy strategy;
}
{
    (
        type = HeavyDBDataType()
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
        [ encoding = HeavyDBEncodingOpt() ]
        {
            list.add(
                SqlDdlNodes.column(s.add(id).end(this), id,
                    type.withEncoding(encoding).withNullable(nullable),
                    defval, strategy));
        }
    |
        { list.add(id); }
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
HeavyDBOptionsMap OptionsOpt() :
{
    HeavyDBOptionsMap optionMap = new HeavyDBOptionsMap();
    Pair<String, SqlNode> optionPair = null;
}
{
  <LPAREN>
  optionPair = KVOption()
  { HeavyDBOptionsMap.add(optionMap, optionPair.getKey(), optionPair.getValue()); }
  (
    <COMMA>
    optionPair = KVOption()
    { HeavyDBOptionsMap.add(optionMap, optionPair.getKey(), optionPair.getValue()); }
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
SqlDdl SqlCreateTable(Span s) :
{
    boolean temporary = false;
    boolean ifNotExists = false;
    final SqlIdentifier id;
    SqlNodeList tableElementList = null;
    HeavyDBOptionsMap withOptions = null;
    SqlNode query = null;
}
{
    [<TEMPORARY> {temporary = true; }]
    <TABLE> 
    ifNotExists = IfNotExistsOpt() 
    id = SimpleIdentifier()
    ( 
        tableElementList = TableElementList()
    |
        <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
    )
    [ <WITH> withOptions = OptionsOpt() ]
    {
        return SqlDdlNodes.createTable(s.end(this), temporary, ifNotExists, id,
            tableElementList, withOptions, query);
    }
}

/**
 * Parses a CREATE statement.
 *
 * This broke away from the default Calcite implementation because we do
 * not allow the the optional "OR REPLACE" clause. 
 *
 */
SqlDdl SqlCustomCreate(Span s) :
{
    final SqlDdl create;
}
{
    <CREATE>
    (
        LOOKAHEAD(1) create = SqlCreateDB(s)
        |
        LOOKAHEAD(1) create = SqlCreateTable(s)
        |
        LOOKAHEAD(1) create = SqlCreateView(s)
        |
        LOOKAHEAD(1) create = SqlCreateRole(s)
        |
        LOOKAHEAD(1) create = SqlCreateDataframe(s)
        |
        LOOKAHEAD(1) create = SqlCreatePolicy(s)
        |
        LOOKAHEAD(1) create = SqlCreateServer(s)
        |
        LOOKAHEAD(1) create = SqlCreateForeignTable(s)
        |
        LOOKAHEAD(2) create = SqlCreateUserMapping(s)
        |
        LOOKAHEAD(2) create = SqlCreateUser(s)
    )
    {
        return create;
    }
}

/**
 * Parses a DROP statement.
 *
 *  This broke away from the default Calcite implementation because it was
 *  necessary to use LOOKAHEAD(2) for the USER commands in order for them 
 *  to parse correctly.
 *
 */
SqlDdl SqlCustomDrop(Span s) :
{
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
        LOOKAHEAD(1) drop = SqlDropModel(s)
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
        |
        LOOKAHEAD(2) drop = SqlDropPolicy(s)
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
        columnList = RepeatedAlterColumnList()
        {
            sqlAlterTableBuilder.addAlterColumnList(columnList);
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

/*
 * Insert into table(s) using one of the following forms:
 *
 * 1) INSERT INTO <table_name> [columns] <values>
 * 2) INSERT INTO <table_name> [columns] <select>
 */
SqlNode SqlInsertIntoTable(Span s) :
{
    final List<SqlLiteral> keywords = new ArrayList<SqlLiteral>();
    final SqlNodeList keywordList = new SqlNodeList(SqlParserPos.ZERO);
    SqlNode table;
    SqlNode source;
    SqlNodeList columnList = null;
}
{
    <INSERT> <INTO> table = CompoundIdentifier()
    [
        LOOKAHEAD(3)
        columnList = ParenthesizedSimpleIdentifierList()
    ]
    (
        source = TableConstructor() {
             return new SqlInsertValues(s.end(this), table, source, columnList);
        }
    |
        source = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY) {
            return new SqlInsertIntoTable(s.end(this), table, source, columnList);
        }
    )
}

/*
 * Dump a table using the following syntax:
 *
 * DUMP TABLE <tableName> TO <filePath> [WITH options]
 *
 */
SqlDdl SqlDumpTable(Span s) :
{
    final SqlIdentifier tableName;
    HeavyDBOptionsMap withOptions = null;
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
 * RESTORE TABLE <tableName> FROM <path> [WITH options]
 *
 */
SqlDdl SqlRestoreTable(Span s) :
{
    final SqlIdentifier tableName;
    HeavyDBOptionsMap withOptions = null;
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
    HeavyDBOptionsMap withOptions = null;   
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
 * Evaluate a model using the following syntax:
 *
 * EVALUATE MODEL <modelName> ON <query>
 *
 */
SqlDdl SqlEvaluateModel(Span s) :
{
    final SqlIdentifier modelName;
    SqlNode query = null;
}
{
    <EVALUATE>
    <MODEL>
    modelName = CompoundIdentifier()
    [ <ON> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY) ]
    {
        return new SqlEvaluateModel(s.end(this), modelName.toString(), query);
    }
}

/*
 * Create a view using the following syntax:
 *
 * CREATE VIEW [ IF NOT EXISTS ] <view_name> [(columns)] AS <query>
 */
SqlDdl SqlCreateView(Span s) :
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
        return SqlDdlNodes.createView(s.end(this), ifNotExists, id, columnList,
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

SqlCreate SqlCreateModel(Span s, boolean replace) :
{
    final boolean ifNotExists;
    final SqlIdentifier modelType;
    final SqlIdentifier id;
    HeavyDBOptionsMap withOptions = null;   
    SqlNode query = null;
}
{
    <MODEL> ifNotExists = IfNotExistsOpt() id = CompoundIdentifier()
    <OF> <TYPE> modelType = CompoundIdentifier()
    <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
    [ <WITH> withOptions = OptionsOpt() ]
    {
        return SqlDdlNodes.createModel(s.end(this), replace, ifNotExists, modelType, id,
            withOptions, query);
    }
}

/*
 * Drop a model using the following syntax:
 *
 * DROP MODEL [ IF EXISTS ] <model_name>
 */
SqlDdl SqlDropModel(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier modelName;
}
{
    <MODEL>
    ifExists = IfExistsOpt()
    modelName = CompoundIdentifier()
    {
        return new SqlDropModel(s.end(this), ifExists, modelName.toString());
    }
}

/*
 * Create a database using the following syntax:
 *
 * CREATE DATABASE ...
 */
SqlDdl SqlCreateDB(Span s) :
{
    final boolean ifNotExists;
    final SqlIdentifier dbName;
    HeavyDBOptionsMap dbOptions = null;
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
 * Alter an existing database using one of two currently supported variants with
 * the following syntax:
 *
 * ALTER DATABASE <database_name> OWNER TO <new_owner>
 * ALTER DATABASE <database_name> RENAME TO <new_database_name>
 */
SqlDdl SqlAlterDatabase(Span s) :
{
    SqlAlterDatabase.Builder sqlAlterDatabaseBuilder = new SqlAlterDatabase.Builder();
    SqlIdentifier databaseName;
    SqlIdentifier sqlIdentifier;
}
{
    <ALTER> <DATABASE>
    databaseName=CompoundIdentifier()
    {
        sqlAlterDatabaseBuilder.setDatabaseName(databaseName.toString());
    }
    (
        <OWNER> <TO> 
        sqlIdentifier=CompoundIdentifier()
        { 
            sqlAlterDatabaseBuilder.setNewOwner(sqlIdentifier.toString());
            sqlAlterDatabaseBuilder.setAlterType(SqlAlterDatabase.AlterType.CHANGE_OWNER);
        }
    |
        <RENAME> <TO>
        sqlIdentifier=CompoundIdentifier()
        {
            sqlAlterDatabaseBuilder.setNewDatabaseName(sqlIdentifier.toString());
            sqlAlterDatabaseBuilder.setAlterType(SqlAlterDatabase.AlterType.RENAME_DATABASE);
        }
    )
    {
        sqlAlterDatabaseBuilder.setPos(s.end(this));
        return sqlAlterDatabaseBuilder.build();
    }
}


/*
 * Create a user using the following syntax:
 *
 * CREATE USER ["]<name>["] (<property> = value,...);
 */
SqlDdl SqlCreateUser(Span s) :
{
    final SqlIdentifier userName;
    HeavyDBOptionsMap userOptions = null;
}
{
    <USER> 
    userName = HyphenatedCompoundIdentifier()
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
    final boolean ifExists;
    final SqlIdentifier userName;
}
{
    <USER>
    ifExists = IfExistsOpt()
    userName = HyphenatedCompoundIdentifier()
    {
        return new SqlDropUser(s.end(this), ifExists, userName.toString());
    }
}

/*
 * Set a comment on a database object using:
 *
 * COMMENT ON (TABLE | COLUMN)  <object_name> IS (<string_literal> | NULL);
 *
 */
SqlDdl SqlComment(Span s) :
{
    SqlComment.Builder builder =
        new SqlComment.Builder();
    SqlIdentifier tableName;
    SqlIdentifier columnName;
    final SqlNode value;
}
{
    <COMMENT> <ON> 
    (
       <TABLE>
       { builder.setTableType(); }
       tableName=SimpleIdentifier()
       { builder.setTableName(tableName.toString()); }
       |
       <COLUMN>
       { builder.setColumnType(); }
       tableName=SimpleIdentifier()
       { builder.setTableName(tableName.toString()); }
       <DOT>
       columnName=SimpleIdentifier()
       { builder.setColumnName(columnName.toString()); }
    )
    <IS> 
    (
       <NULL>
       { builder.setToNull(); }
       |
       value = StringLiteral()
       { builder.setComment((new HeavySqlSanitizedString(value)).toString()); }
    )
    {
        builder.setPos(s.end(this));
        return builder.build();
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
    HeavyDBOptionsMap userOptions = null;
}
{
    <ALTER>
    <USER>
    userName = HyphenatedCompoundIdentifier()
    (
        <RENAME>
        <TO>
        newUserName = HyphenatedCompoundIdentifier()
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

/**
 * Parses a compound identifier with hyphens/minus as valid characters 
 *     for consistency with the old Bison parser
 *
 * A (better?) solution might be to rewrite the token generator in Calcite to allow hypens, 
 * but this is a very narrow use case, so allow hypens by recreating the full hyphenated 
 * token by reconstructing from the individual token - pieces. A by-product of this 
 * approach is that spaces can now appear before/after the hypens, so test for them 
 * and trigger failure by not creating the compoundIdentifier.
 *
 */
public void HyphenatedIdentifierSegment(List<String> nameList, List<SqlParserPos> posList) :
{
    // use local lists/variables as it'll collapse them before returning
    final List<String> localNameList = new ArrayList<String>();
    final List<SqlParserPos> localPosList = new ArrayList<SqlParserPos>();
    boolean trailingHyphen = false;
}
{
    IdentifierSegment(localNameList, localPosList)
    (
        LOOKAHEAD(2)
        <MINUS>
        IdentifierSegment(localNameList, localPosList)
    )*
    (
        <MINUS>
        { trailingHyphen = true; }
    )?
    {
        if(localNameList.size() > 0 && localPosList.size() > 0) {
            // Rebuild the hyphenated name from the identifiers
            String recreatedName = "";
            SqlParserPos pos = SqlParserPos.ZERO;
            int startCol = 0;

            for (int i = 0; i < localNameList.size() && i < localPosList.size(); i++){
                if(i == 0){
                    startCol = localPosList.get(i).getColumnNum();
                } else {
                    if((localPosList.get(i).getColumnNum() - localPosList.get(i-1).getEndColumnNum()) != 2){
                        // detected error or spaces around the hypen ... just fail outright
                        return;                       
                    }
                    recreatedName += "-";
                }
                recreatedName += localNameList.get(i);
                pos = localPosList.get(i);
            }
            if(trailingHyphen){
                recreatedName += "-";
            }

            SqlParserPos recreatedPos = new SqlParserPos(pos.getLineNum(), startCol, pos.getLineNum(), pos.getEndColumnNum());

            // add the collapsed name/pos to the incoming/outgoing Lists
            nameList.add(recreatedName);
            posList.add(recreatedPos);
        }
    }
}

/**
 * Parses a compound identifier.
 *
 * Copied (almost) verbatum from Calcite's CompoundIdentifier(), except that
 *    it'll call HyphenatedIdentifierSegment() instead of IdentifierSegment()
 */
SqlIdentifier HyphenatedCompoundIdentifier() :
{
    final List<String> nameList = new ArrayList<String>();
    final List<SqlParserPos> posList = new ArrayList<SqlParserPos>();
    boolean star = false;
}
{
    HyphenatedIdentifierSegment(nameList, posList)
    (
        LOOKAHEAD(2)
        <DOT>
        HyphenatedIdentifierSegment(nameList, posList)
    )*
    (
        LOOKAHEAD(2)
        <DOT>
        <STAR> {
            star = true;
            nameList.add("");
            posList.add(getPos());
        }
    )?
    {
        SqlParserPos pos = SqlParserPos.sum(posList);
        if (star) {
            return SqlIdentifier.star(nameList, pos, posList);
        } else if (nameList.size() > 0) {
            return new SqlIdentifier(nameList, null, pos, posList);
        }
    }
}

/*
 * Create a role using the following syntax:
 *
 * CREATE ROLE <role_name>
 *
 */
SqlDdl SqlCreateRole(Span s) :
{
    final SqlIdentifier role;
}
{
    <ROLE> role = HyphenatedCompoundIdentifier()
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
    final boolean ifExists;
    final SqlIdentifier role;
}
{
    <ROLE> 
    ifExists = IfExistsOpt()
    role = HyphenatedCompoundIdentifier()
    {
        return new SqlDropRole(s.end(this), ifExists, role.toString());
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
    |   LOOKAHEAD(2) <CREATE> <MODEL> { type = "CREATE MODEL"; }
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
 * Create a table using the following syntax:
 *
 *		CREATE DATAFRAME table '(' base_table_element_commalist ')' FROM STRING opt_with_option_list
 */
SqlDdl SqlCreateDataframe(Span s) :
{
    SqlIdentifier name;
    SqlNodeList elementList = null;
    SqlNode filePath = null;
    HeavyDBOptionsMap dataframeOptions = null;
}
{
    <DATAFRAME> 
    name = CompoundIdentifier()
    elementList = TableElementList()
    <FROM>
    filePath = StringLiteral()
    [ <WITH> dataframeOptions = OptionsOpt() ]
    {
        return new SqlCreateDataframe(s.end(this), name, elementList, filePath, dataframeOptions);
    }
}

/*
 * CREATE POLICY
 */
SqlDdl SqlCreatePolicy(Span s) :
{
    SqlIdentifier columnName = null;
    SqlIdentifier granteeName = null;
    SqlNodeList valuesList = null;
}
{
    <POLICY>
    <ON>
    <COLUMN>
    columnName = CompoundIdentifier()
    <TO>
    granteeName = CompoundIdentifier()
    <VALUES>
    (
        valuesList = ParenthesizedLiteralOptionCommaList()
    )
    {
        return new SqlCreatePolicy(s.end(this), columnName.names, valuesList, granteeName);
    }
}

/*
 * DROP POLICY
 */
SqlDrop SqlDropPolicy(Span s) :
{
    SqlIdentifier columnName = null;
    SqlIdentifier granteeName = null;
}
{
    <POLICY>
    <ON>
    <COLUMN>
    columnName = CompoundIdentifier()
    <FROM>
    granteeName = CompoundIdentifier()
    {
        return new SqlDropPolicy(s.end(this), columnName.names, granteeName);
    }
}

/*
 * Parse the ALL keyphrase.
 *
 * [ ALL ]
 */
boolean All() :
{
}
{
    <ALL> { return true; }
    |
    { return false; }
}

/*
 * Reassign owned database objects using the following syntax:
 *
 * REASSIGN [ALL] OWNED BY <old_owner>, <old_owner>, ... TO <new_owner>
 */
SqlDdl SqlReassignOwned(Span s) :
{
    SqlIdentifier userName = null;
    List<String> oldOwners = null;
    final boolean all;
}
{
    <REASSIGN>
    all = All()
    <OWNED> <BY>
    userName = HyphenatedCompoundIdentifier()
    {
        oldOwners = new ArrayList<String>();
        oldOwners.add(userName.toString());
    }
    (
        <COMMA>
        userName = HyphenatedCompoundIdentifier()
        {
            oldOwners.add(userName.toString());
        }
    )*
    <TO>
    userName = HyphenatedCompoundIdentifier()
    {
        return new SqlReassignOwned(s.end(this), oldOwners, userName.toString(), all);
    }
}

/*
 * Validate the system using the following syntax:
 *
 *		VALIDATE [CLUSTER [WITH options]]
 */
SqlDdl SqlValidateSystem(Span s) :
{
    HeavyDBOptionsMap validateOptions = null;
    String type = "";
}
{
    <VALIDATE> 
    [<CLUSTER> {type = "CLUSTER";}
        [<WITH> validateOptions = OptionsOpt() ]]
    {
        return new SqlValidateSystem(s.end(this), type, validateOptions);
    }
}

/*
 * Validate the system using the following syntax:
 *
 *	COPY table FROM <STRING> opt_with_option_list
 *	COPY '(' <FWDSTR> ')' TO <STRING> opt_with_option_list
 *     -> FWDSTR starts with 'select' or 'with' only 
 *           (WITH d2 AS (SELECT x ....))
 *
 */
SqlDdl SqlCopyTable(Span s) :
{
    HeavyDBOptionsMap copyOptions = null;
    SqlNode filePath = null;
    SqlIdentifier table = null;
    SqlNode query = null;
}
{
    <COPY> 
    (
        table = CompoundIdentifier()
        <FROM>  
    |
        <LPAREN> 
        query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
        <RPAREN> 
        <TO> 
    )
    filePath = StringLiteral()
    [<WITH> copyOptions = OptionsOpt()]
    {
        if(table != null){
            return new SqlCopyTable(s.end(this), table.toString(), filePath.toString(), copyOptions);
        } else {
            return new SqlExportQuery(s.end(this), query, filePath.toString(), copyOptions);
        }
    }
}

/** Parses an ARRAY constructor */
SqlNode CurlyArrayConstructor() :
{
    SqlNodeList args;
    SqlNode e;
    final Span s;
}
{
    <LBRACE> { s = span(); }
    (
        args = ExpressionCommaList(s, ExprContext.ACCEPT_NON_QUERY)
    |
        { args = SqlNodeList.EMPTY; }
    )
    <RBRACE>
    {
        return SqlStdOperatorTable.ARRAY_VALUE_CONSTRUCTOR.createCall(
            s.end(this), args.getList());
    }
}

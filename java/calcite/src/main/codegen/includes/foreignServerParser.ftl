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
 * Create a new foreign server using the following syntax:
 *
 * CREATE SERVER [IF NOT EXISTS] <server_name>
 *   FOREIGN DATA WRAPPER <foreign_data_wrapper_name>
 *   WITH ( <option> = <value> [, ... ] )
 */
SqlCreate SqlCreateServer(Span s, boolean replace) :
{
    SqlCreateServer.Builder sqlCreateServerBuilder = new SqlCreateServer.Builder();
    SqlIdentifier sqlIdentifier;
    boolean ifNotExists;
}
{
    <SERVER>
    ifNotExists=IfNotExistsOpt()
    {
        sqlCreateServerBuilder.setIfNotExists(ifNotExists);
    }
    sqlIdentifier=CompoundIdentifier()
    {
        sqlCreateServerBuilder.setServerName(sqlIdentifier.toString());
    }
    <FOREIGN> <DATA> <WRAPPER>
    sqlIdentifier=CompoundIdentifier()
    {
        sqlCreateServerBuilder.setDataWrapper(sqlIdentifier.toString());
    }
    [ Options(sqlCreateServerBuilder) ]
    {
        sqlCreateServerBuilder.setPos(s.end(this));
        return sqlCreateServerBuilder.build();
    }
}

/*
 * Drop a foreign server using the following syntax:
 *
 * DROP SERVER [ IF EXISTS ] <server_name>
 */
SqlDrop SqlDropServer(Span s, boolean replace) :
{
    final boolean ifExists;
    final SqlIdentifier serverName;
}
{
    <SERVER>
    ifExists = IfExistsOpt()
    serverName = CompoundIdentifier()
    {
        return new SqlDropServer(s.end(this), ifExists, serverName.toString());
    }
}

SqlDdl SqlShowForeignServers(Span s) :
{
    SqlShowForeignServers.Builder sqlShowForeignServersBuilder = new SqlShowForeignServers.Builder();
}
{
    <SHOW> <SERVERS> 
    
    [
        WhereClause(sqlShowForeignServersBuilder)
    ]

    {
        sqlShowForeignServersBuilder.setPos(s.end(this));
        return sqlShowForeignServersBuilder.build();
    }
}


/*
 * Parse the IF NOT EXISTS keyphrase.
 *
 * [ IF NOT EXISTS ]
 */
boolean IfNotExistsOpt() :
{
}
{
    <IF> <NOT> <EXISTS> { return true; }
|
    { return false; }
}

/*
 * Parse the IF EXISTS keyphrase.
 *
 * [ IF EXISTS ]
 */
boolean IfExistsOpt() :
{
}
{
    <IF> <EXISTS> { return true; }
|
    { return false; }
}

/*
 * Parse the IF NOT EXISTS keyphrase.
 *
 * WITH ( <option> = <value> [, ... ] )
 */
void Options(SqlCreateServer.Builder sqlCreateServerBuilder) :
{
}
{
    <WITH> <LPAREN>
    Option(sqlCreateServerBuilder)
    (
        <COMMA>
        Option(sqlCreateServerBuilder)
    )*
    <RPAREN>
}

/*
 * Parse the IF NOT EXISTS keyphrase.
 *
 * WITH ( <option> = <value> [, ... ] )
 */
void Option(SqlCreateServer.Builder sqlCreateServerBuilder) :
{
    final SqlIdentifier attribute;
    final SqlNode value;
}
{
    attribute = CompoundIdentifier()
    <EQ>
    value = Literal()
    {
        sqlCreateServerBuilder.addOption(attribute.toString(), value.toString());
    }
}

void WhereClause(SqlShowForeignServers.Builder sqlShowForeignServersBuilder) :
{
    SqlFilter.Chain chain;
}
{
    <WHERE>
    Predicate(sqlShowForeignServersBuilder, null)
    (
        (
        <AND> { chain = SqlFilter.Chain.AND; } | <OR> { chain = SqlFilter.Chain.OR; }
        )
        Predicate(sqlShowForeignServersBuilder, chain)
    )*
}

void Predicate(SqlShowForeignServers.Builder sqlShowForeignServersBuilder, SqlFilter.Chain chain) :
{
    final SqlIdentifier attribute;
    final SqlFilter.Operation operation;
    final SqlNode value;    
}
{ 
    attribute=CompoundIdentifier()
    (
        <EQ> { operation = SqlFilter.Operation.EQUALS; } | <LIKE> { operation = SqlFilter.Operation.LIKE; }
    )
    value = Literal()
    {
        sqlShowForeignServersBuilder.addFilter(attribute.toString(), value.toString(), operation, chain);
    }
}

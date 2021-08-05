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
 * Create a new user mapping using the following syntax:
 *
 * CREATE USER MAPPING [IF NOT EXISTS] FOR { <user> | CURRENT_USER | PUBLIC }
 *   SERVER <server_name>
 *   WITH ( <option> = <value> [, ... ] )
 */
SqlCreate SqlCreateUserMapping(Span s, boolean replace) :
{
    SqlCreateUserMapping.Builder sqlCreateUserMappingBuilder = new SqlCreateUserMapping.Builder();
    final String user;
    final SqlIdentifier serverName;
    final boolean ifNotExists;
}
{
    <USER> <MAPPING>
    ifNotExists = IfNotExistsOpt()
    {
        sqlCreateUserMappingBuilder.setIfNotExists(ifNotExists);
    }
    user = ForUser()
    {
        sqlCreateUserMappingBuilder.setUser(user);
    }
    <SERVER>
    serverName = CompoundIdentifier()
    {
        sqlCreateUserMappingBuilder.setServerName(serverName.toString());
    }
    <WITH>
    Options(sqlCreateUserMappingBuilder)
    {
        sqlCreateUserMappingBuilder.setPos(s.end(this));
        return sqlCreateUserMappingBuilder.build();
    }
}

/*
 * Drop a user mapping using the following syntax:
 *
 * DROP USER MAPPING [IF EXISTS] FOR { <user> | CURRENT_USER | PUBLIC }
 *     SERVER <server_name>
 */
SqlDrop SqlDropUserMapping(Span s) :
{
    final boolean ifExists;
    final String user;
    final SqlIdentifier serverName;
}
{
    <USER> <MAPPING>
    ifExists = IfExistsOpt()
    user = ForUser()
    <SERVER>
    serverName = CompoundIdentifier()
    {
        return new SqlDropUserMapping(s.end(this), ifExists, user, serverName.toString());
    }
}

/*
 * Parses the "FOR { <user> | CURRENT_USER | PUBLIC }" phrase
 */
String ForUser() :
{
    final SqlIdentifier sqlIdentifier;
}
{
    <FOR>
    (
        <CURRENT_USER>
        {
            return "CURRENT_USER";
        }
    |
        // "PUBLIC" will also be captured by CompoundIdentifier()
        sqlIdentifier = CompoundIdentifier()
        {
            return sqlIdentifier.toString();
        }
    )
}

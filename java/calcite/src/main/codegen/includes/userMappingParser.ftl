<#--
 SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0
-->

/*
 * Create a new user mapping using the following syntax:
 *
 * CREATE USER MAPPING [IF NOT EXISTS] FOR { <user> | CURRENT_USER | PUBLIC }
 *   SERVER <server_name>
 *   WITH ( <option> = <value> [, ... ] )
 */
SqlCreate SqlCreateUserMapping(Span s) :
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

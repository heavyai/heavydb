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
 *
 * SHOW USER DETAILS
 *
 */

SqlDdl SqlShowUserDetails(Span s) :
{
    SqlIdentifier userName = null;
    List<String> userNames = null;
}
{
    <USER>
    <DETAILS>
    [
        userName = CompoundIdentifier()
        {
            userNames = new ArrayList<String>();
            userNames.add(userName.toString());
        }
        (
            <COMMA>
            userName = CompoundIdentifier()
            {
                userNames.add(userName.toString());
            }
        )*
    ]
    {
        return new SqlShowUserDetails(s.end(this), userNames);
    }
}

/*
 *
 * SHOW USER SESSIONS
 *
 */

SqlDdl SqlShowUserSessions(Span s) :
{
}
{
    <USER>
    <SESSIONS>
    {
        return new SqlShowUserSessions(s.end(this));
    }
}

/*
 * Show existing tables using the following syntax:
 *
 * SHOW TABLES
 */
SqlDdl SqlShowTables(Span s) :
{
}
{
    <TABLES>
    {
        return new SqlShowTables(s.end(this));
    }
}


/*
 * Show databases using the following syntax:
 *
 * SHOW DATABASES
 */
SqlDdl SqlShowDatabases(Span s) :
{
}
{
    <DATABASES>
    {
        return new SqlShowDatabases(s.end(this));
    }
}

/*
 * Show query info using the following syntax:
 *
 * SHOW QUERIES
 */

SqlDdl SqlShowQueries(Span s) :
{
}
{
    <QUERIES>
    {
        return new SqlShowQueries(s.end(this));
    }
}

SqlDdl SqlShowDiskCacheUsage(Span s) : {
    SqlIdentifier tableName = null;
    List<String> tableNames = new ArrayList<String>();
}
{
    <DISK> <CACHE> <USAGE>
    (
        tableName = CompoundIdentifier()
        { tableNames.add(tableName.toString()); }
        (
            <COMMA>
            tableName = CompoundIdentifier()
            { tableNames.add(tableName.toString()); }
        )*
        { return new SqlShowDiskCacheUsage(s.end(this), tableNames); }
    |
        { return new SqlShowDiskCacheUsage(s.end(this)); }
    )
}

/*
 * Show table details using the following syntax:
 *
 * SHOW TABLE DETAILS [<table_name>, <table_name>, ...]
 */
SqlDdl SqlShowTableDetails(Span s) :
{
    SqlIdentifier tableName = null;
    List<String> tableNames = null;
}
{
    <TABLE> <DETAILS>
    [
        tableName = CompoundIdentifier()
        {
            tableNames = new ArrayList<String>();
            tableNames.add(tableName.toString());
        }
        (
            <COMMA>
            tableName = CompoundIdentifier()
            {
                tableNames.add(tableName.toString());
            }
        )*
    ]
    {
        return new SqlShowTableDetails(s.end(this), tableNames);
    }
}

/*
 * SHOW CREATE TABLE <table_name>
 */
SqlDdl SqlShowCreateTable(Span s) :
{
    SqlIdentifier tableName = null;
}
{
    <CREATE> <TABLE>
    tableName = CompoundIdentifier()
    {
        return new SqlShowCreateTable(s.end(this), tableName.toString());
    }
}

/*
 * SHOW [EFFECTIVE] ROLES [username]
 */
SqlDdl SqlShowRoles(Span s) :
{
    SqlIdentifier userName = null;
    boolean effective = false;
}
{
    [<EFFECTIVE> {effective = true;}]
    <ROLES>
    [userName = CompoundIdentifier()]
    {
        String u = "";
        if (userName != null) {
          u = userName.toString();
        }
        return new SqlShowRoles(s.end(this), u, effective);
    }
}

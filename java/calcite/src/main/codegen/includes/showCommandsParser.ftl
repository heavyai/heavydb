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
 * <SHOW> <USER> <DETAILS>
 * <SHOW> <USER> <SESSIONS>
 *
 */

SqlDdl SqlShowUser(Span s) :
{
    SqlIdentifier userName = null;
    List<String> userNames = null;
}
{
    <SHOW> <USER>
    (
        <SESSIONS>
        {
            return new SqlShowUserSessions(s.end(this));
        }
    |
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
    )
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
    <SHOW> <TABLES>
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
    <SHOW> <DATABASES>
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
    <SHOW> <QUERIES>
    {
        return new SqlShowQueries(s.end(this));
    }
}

SqlDdl SqlShowDiskCacheUsage(Span s) : {
    SqlIdentifier tableName = null;
    List<String> tableNames = new ArrayList<String>();
}
{
    <SHOW> <DISK> <CACHE> <USAGE>
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
    <SHOW> <TABLE> <DETAILS>
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

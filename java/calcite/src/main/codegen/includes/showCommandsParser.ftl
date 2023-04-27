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

/**
 * Parses a SHOW statement.
 *
 *  These SHOW statements used to be listed directly in statementParserMethods in
 *  config.fmpp but those all default to LOOKAHEAD(2), hardcoded in Calcite's Parser.jj
 *  source code. Some SHOW statements needed LOOKAHEAD(3) there so all SHOW statements
 *  had to be moved into this two-stage SqlCustomShow definition (where they only have
 *  LOOKAHEAD(1) or LOOKAHEAD(2) because the <SHOW> is no longer part of that count).
 */
SqlDdl SqlCustomShow(Span s) :
{
    final SqlDdl show;
}
{
    <SHOW>
    (
        LOOKAHEAD(3) show = SqlShowUserDetails(s)
        |
        LOOKAHEAD(2) show = SqlShowUserSessions(s)
        |
        LOOKAHEAD(1) show = SqlShowTables(s)
        |
        LOOKAHEAD(2) show = SqlShowTableDetails(s)
        |
        LOOKAHEAD(1) show = SqlShowFunctions(s)
        |
        LOOKAHEAD(2) show = SqlShowRuntimeFunctions(s)
        |
        LOOKAHEAD(2) show = SqlShowTableFunctions(s)
        |
        LOOKAHEAD(3) show = SqlShowRuntimeTableFunctions(s)
        |
        LOOKAHEAD(1) show = SqlShowModels(s)
        |
        LOOKAHEAD(2) show = SqlShowModelDetails(s)
        |
        LOOKAHEAD(2) show = SqlShowModelFeatureDetails(s)
        |
        LOOKAHEAD(1) show = SqlShowDatabases(s)
        |
        LOOKAHEAD(1) show = SqlShowForeignServers(s)
        |
        LOOKAHEAD(1) show = SqlShowQueries(s)
        |
        LOOKAHEAD(1) show = SqlShowDiskCacheUsage(s)
        |
        LOOKAHEAD(1) show = SqlShowCreate(s)
        |
        LOOKAHEAD(2) show = SqlShowRoles(s)
        |
        LOOKAHEAD(2) show = SqlShowPolicies(s)
        |
        LOOKAHEAD(1) show = SqlShowDataSources(s)
    )
    {
        return show;
    }
}

/*
 *
 * SHOW USER DETAILS
 *
 */

SqlDdl SqlShowUserDetails(Span s) :
{
    SqlIdentifier userName = null;
    List<String> userNames = null;
    boolean all = false;
}
{
    [<ALL> {all = true;}]
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
        return new SqlShowUserDetails(s.end(this), userNames, all);
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

/*
 * Show existing models using the following syntax:
 *
 * SHOW MODELS
 */
 
SqlDdl SqlShowModels(Span s) :
{
}
{
    <MODELS>
    {
        return new SqlShowModels(s.end(this));
    }
}

/*
 * Show model details using the following syntax:
 *
 * SHOW MODEL DETAILS [<model_name>, <model_name>, ...]
 */
 
SqlDdl SqlShowModelDetails(Span s) :
{
    SqlIdentifier modelName = null;
    List<String> modelNames = null;
}
{
    <MODEL> <DETAILS>
    [
        modelName = CompoundIdentifier()
        {
            modelNames = new ArrayList<String>();
            modelNames.add(modelName.toString());
        }
        (
            <COMMA>
            modelName = CompoundIdentifier()
            {
                modelNames.add(modelName.toString());
            }
        )*
    ]
    {
        return new SqlShowModelDetails(s.end(this), modelNames);
    }
}

/*
 * Show model feature details using the following syntax:
 *
 * SHOW MODEL FEATURE DETAILS <model_name>
 */
 
SqlDdl SqlShowModelFeatureDetails(Span s) :
{
    SqlIdentifier modelName = null;
}
{
    <MODEL> <FEATURE> <DETAILS>
    (
        modelName = CompoundIdentifier()
    )
    {
        return new SqlShowModelFeatureDetails(s.end(this), modelName.toString());
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
 * SHOW FUNCTIONS [ DETAILS <scalar_fn_name>, <scalar_fn_name>, ...]
 */
SqlDdl SqlShowFunctions(Span s) :
{
    SqlIdentifier UdfName = null;
    List<String> ScalarFnNames = null;
}
{
    <FUNCTIONS>
    [
        <DETAILS>
        UdfName = CompoundIdentifier()
        {
            ScalarFnNames = new ArrayList<String>();
            ScalarFnNames.add(UdfName.toString());
        }
        (
            <COMMA>
            UdfName = CompoundIdentifier()
            {
                ScalarFnNames.add(UdfName.toString());
            }
        )*
    ]
    {
        return new SqlShowFunctions(s.end(this), ScalarFnNames);
    }
}

/*
 * SHOW RUNTIME FUNCTIONS
 */
SqlDdl SqlShowRuntimeFunctions(Span s) :
{
}
{
    <RUNTIME> <FUNCTIONS>
    {
        return new SqlShowRuntimeFunctions(s.end(this));
    }
}

/*
 * SHOW TABLE FUNCTIONS [ DETAILS <table_function_name>, <table_function_name>, ...]
 */
SqlDdl SqlShowTableFunctions(Span s) :
{
    SqlIdentifier tfName = null;
    List<String> tfNames = null;
}
{
    <TABLE> <FUNCTIONS>
    [
        <DETAILS>
        tfName = CompoundIdentifier()
        {
            tfNames = new ArrayList<String>();
            tfNames.add(tfName.toString());
        }
        (
            <COMMA>
            tfName = CompoundIdentifier()
            {
                tfNames.add(tfName.toString());
            }
        )*
    ]
    {
        return new SqlShowTableFunctions(s.end(this), tfNames);
    }
}

/*
 * SHOW RUNTIME TABLE FUNCTIONS
 */
SqlDdl SqlShowRuntimeTableFunctions(Span s) :
{
}
{
    <RUNTIME> <TABLE> <FUNCTIONS>
    {
        return new SqlShowRuntimeTableFunctions(s.end(this));
    }
}

/*
 * SHOW CREATE TABLE <table_name>
 */
SqlDdl SqlShowCreate(Span s) :
{
    SqlIdentifier name = null;
}
{
    <CREATE>
    (
        <TABLE>
        name = CompoundIdentifier()
        {
            return new SqlShowCreateTable(s.end(this), name.toString());
        }
    |
        <SERVER>
        name = CompoundIdentifier()
        {
            return new SqlShowCreateServer(s.end(this), name.toString());
        }
    )
}

/*
 * SHOW [EFFECTIVE] ROLES [username]
 */
SqlDdl SqlShowRoles(Span s) :
{
    boolean effective = false;
    SqlIdentifier userName = null;
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

/*
 * SHOW [EFFECTIVE] POLICIES [username]
 */
SqlDdl SqlShowPolicies(Span s) :
{
    boolean effective = false;
    SqlIdentifier granteeName = null;
}
{
    [<EFFECTIVE> {effective = true;}]
    <POLICIES>
    granteeName = CompoundIdentifier()
    {
        String n;
        if (granteeName != null) {
          n = granteeName.toString();
        } else {
          n = "";
        }
        return new SqlShowPolicies(s.end(this), effective, n);
    }
}

/*
 * Show supported data sources using the following syntax:
 *
 * SHOW SUPPORTED DATA SOURCES
 */
SqlDdl SqlShowDataSources(Span s) :
{
}
{
    <SUPPORTED> <DATA> <SOURCES>
    {
        return new SqlShowDataSources(s.end(this));
    }
}

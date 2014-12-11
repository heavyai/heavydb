#include "SqliteConnector.h" 
#include <iostream>

using namespace std;

int resultCallback(void *connObj, int argc, char ** argv, char **colNames) {
    cout << "callback" << endl;
    SqliteConnector *sqliteConnector = reinterpret_cast<SqliteConnector *> (connObj);
    if (sqliteConnector -> atFirstResult_) {
        sqliteConnector -> numCols_ = argc;
        for (int c = 0; c < argc; ++c) {
            sqliteConnector -> columnNames.push_back(colNames[c]);
        }
        sqliteConnector -> results_.resize(argc);
        sqliteConnector -> atFirstResult_ = false;
    }
    sqliteConnector -> numRows_ = argc;
    for (int c = 0; c < argc; ++c) {
        sqliteConnector -> results_[c].push_back(argv[c]);
    }
    return 0;
}

SqliteConnector::SqliteConnector (const string &dbName, const string &dir) {
    string connectString(dir);
    if (connectString.size() > 0 && connectString[connectString.size()-1] != '/') {
        connectString.push_back('/');
    }
    connectString += dbName;
    int returnCode = sqlite3_open(connectString.c_str(), &db_);
    if (returnCode != SQLITE_OK) {
        string errorMsg (sqlite3_errmsg(db_));
        throw std::runtime_error("Sqlite3 Error: " + errorMsg);
    }
}

void SqliteConnector::query(const std::string &queryString) {
    atFirstResult_ = true;
    numRows_ = 0;
    numCols_ = 0;
    columnNames.clear();
    results_.clear();
    char *errorMsg;
    int returnCode = sqlite3_exec(db_,queryString.c_str(), resultCallback,this,&errorMsg);
    if (returnCode != SQLITE_OK) {
        string errorString ("Sqlite3 Error: ");
        errorString += errorMsg;
        throw std::runtime_error(errorString);
        //string errorMsg (sqlite3_errmsg(db_)):
    }
}

int main () {
    SqliteConnector sqlConnector ("test");
    sqlConnector.query("select * from tables");
    int numRows = sqlConnector.getNumRows();
    int numCols = sqlConnector.getNumCols();
    cout << "Num rows: " << numRows << endl;
    cout << "Num cols: " << numCols << endl;
    for (int r = 0; r < numRows; ++r) {
        cout << sqlConnector.getData<string>(r,0);
        cout << " ";
        cout << sqlConnector.getData<int>(r,1);
        cout << endl;
    }
}


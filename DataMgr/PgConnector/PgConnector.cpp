/**
 * @file        PgConnector.cpp
 * @author      Todd Mostak <todd@map-d.com>
 */

#include "PgConnector.h"
#include <iostream>


using namespace std;


PgConnector::PgConnector (const std::string &dbName, const std::string &userName, const std::string &host, const std::string &port): dbName_(dbName), userName_(userName), host_(host), port_(port),  connectString_("port=" + port + " dbname=" + dbName  + " user=" + userName), conn_(connectString_), numRows_(0), numCols_(0) {}

mapd_err_t PgConnector::query(const std::string &queryString) {
    pqxx::work txn (conn_);
    try {
        pgResults_ = txn.exec(queryString);
    }
    catch (const std::exception &e) {
        cerr << "Postgres " << e.what() << endl;
        txn.abort();
        return MAPD_ERR_PG_FAIL;
    }
    numRows_ = pgResults_.size();
    numCols_ = pgResults_.columns();
    txn.commit();
    return MAPD_SUCCESS;
}

/*
 int main() {
 cout << "Hello" << endl;
 PgConnector pgConnector("mapd","tmostak","127.0.0.1","5432");
 //////////////////mapd_err_t status = pgConnector.query("select * from us_zip");
 mapd_err_t status = pgConnector.query("select table_id, fragment_id, num_rows from fragments");
 cout << "Status: " << status << endl;
 if (status == MAPD_SUCCESS) {
 int numRows = pgConnector.getNumRows();
 for (int r = 0; r != numRows; ++r)
 cout << pgConnector.getData <int> (r,0) << " " << pgConnector.getData<int> (r,1) << " " << pgConnector.getData<int> (r,2) << endl;
 }
 
 
 };
 */


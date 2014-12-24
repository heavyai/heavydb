/**
 * @file		PgConnector.h
 * @author		Todd Mostak <todd@map-d.com>
 */

#ifndef PG_CONNECTOR
#define PG_CONNECTOR
#include "../../Shared/errors.h"
#include <string> 
#include <vector>
#include <pqxx/pqxx>
#include <assert.h>

//typedef std::vector <std::vector <void *> > ResultSet; 

class PgConnector {
    public:

        PgConnector (const std::string &dbName, const std::string &userName, const std::string &host = "localhost", const std::string &port = "5432") ;
        ~PgConnector () {
            conn_.disconnect();
        }
        mapd_err_t query(const std::string &queryString);


        template <typename T> T getData(const int row, const int col) {
            assert (row < numRows_);
            assert (col < numCols_);
            return pgResults_[row][col].as<T>(); 
        }

        inline size_t getNumRows() {return numRows_;}
        inline size_t getNumCols() {return numCols_;}

    private:
        std::string dbName_;
        std::string userName_;
        std::string port_;
        std::string host_;
        std::string connectString_;
        pqxx::connection conn_;
        pqxx::result pgResults_;
        size_t numRows_;
        size_t numCols_;

};

#endif

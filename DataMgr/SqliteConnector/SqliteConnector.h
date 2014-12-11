/**
 * @file		SqliteConnector.h
 * @author		Todd Mostak <todd@map-d.com>
 */

#ifndef SQLITE_CONNECTOR
#define SQLITE_CONNECTOR

#include <string>
#include <vector>
#include <assert.h>
#include <boost/lexical_cast.hpp>

#include "sqlite3.h"

class SqliteConnector {

    friend int resultCallback(void *connObj, int argc, char ** argv, char **colNames);

    public:
        SqliteConnector (const std::string &dbName, const std::string &dir = ".");
        void query(const std::string &queryString);
        void queryWithCallback(const std::string &queryString);

        inline size_t getNumRows() {return numRows_;}
        inline size_t getNumCols() {return numCols_;}

        template <typename T> T getData(const int row, const int col) {
            assert (row < numRows_);
            assert (col < numCols_);
            return boost::lexical_cast <T> (results_[col][row]);
        }
        std::vector<std::string> columnNames; // make this public for easy access


    private:
        void throwError();

        sqlite3 *db_;
        std::string dbName_;
        bool atFirstResult_;
        std::vector<std::vector<std::string> > results_;
        size_t numCols_;
        size_t numRows_;;

};


#endif // SQLITE_CONNECTOR

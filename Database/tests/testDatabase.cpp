/**
 * @file testDatabase.cpp
 * @author Steven Stewart <steve@map-d.com>
 *
 * This program is used to test the Database class.
 */
#include <iostream>
#include "Database.h"
#include "../../Server/Output/OutputBuffer.h"

int main(int argc, char** argv) {
    std::string tcpPort = "7777";
    int numThreads = 1;
    
    Database *db = new Database_Namespace::Database(tcpPort, numThreads);
    //db->start();
    OutputBuffer outbuf;
    db->processRequest("select * from t0 where (a+b) > 0;", outbuf, true);
    
    db->stop();
    delete db;
    return 0;
}

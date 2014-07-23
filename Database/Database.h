#ifndef DATABASE_H
#define DATABASE_H

#include <string>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include "../Server/TcpServer/TcpServer.h"
//#include "TcpServer.h"

class OutputBuffer;

namespace Database_Namespace {

class Database {

    public:
        Database(const std::string &tcpPort, const int numThreads); 
        void start();
        void stop();

        bool processRequest(const std::string &request, OutputBuffer &outputBuffer);


    private:
        std::string tcpPort_;
        int numThreads_;

        /// The io_service used to perform asynchronous operations
        boost::asio::io_service ioService_;

        /// The signal_set is used to register for process termination notifications.
        boost::asio::signal_set signals_;

        TcpServer_Namespace::TcpServer tcpServer_;

        void registerSignals();
};

} // Database_Namespace


#endif

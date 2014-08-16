/**
 * @file    Database.h
 * @author  Todd Mostak <todd@map-d.com>
 * @brief This file contains the class specification for Database,
 * a singleton that is the central hub that pushes queries through
 * the QueryEngine pipeline.
 */


#ifndef DATABASE_H
#define DATABASE_H

#include <string>

#include "../Server/TcpServer/TcpServer.h"
#include "../DataMgr/Metadata/Catalog.h"

class OutputBuffer;

namespace Database_Namespace {

/**
 * @type Database
 * @brief A singleton that is the central hub where queries are
 * pushed through the QueryEngine pipeline.
 *
 * Could be made into a proper Singleton.
 * Manages an io_service thread pool that can be used
 * for incoming connections.
 * Starts a TcpServer on the port specified in the constructor,
 * with the number of threads specified in the constructor.
 */

class Database {

    public:
   
        /**
         * @brief Constructor takes port for TcpServer and 
         * number of threads for io_service
         * @param tcpPort String version of the port that the 
         * TcpServer should listen to.
         * @param @numThreads How many threads to instanciate 
         * io_service with - currently just used for the TcpServer
         * and signal handling but could also be used for other
         * purposes, like an HTTP Server.
         */

        Database(const std::string &tcpPort, const int numThreads); 

        /**
         * @brief Starts database.
         *
         * Creates io_service thread pool, starts
         * TcpServer, and waits for threads to join (on stop()).
         */

        void start();

        /**
         * @brief Stops Database. 
         *
         * Stops TcpServer and then IoService.  Typically 
         * triggered by a kill signal.
         *
         * @see registerSignals() 
         */

        void stop();

        /**
         * @brief Central function for processing incoming query
         * @param request String-representation of the incoming
         * query.
         * @param outputBuffer OutputBuffer already instansciated
         * by TcpConnection where results (or errors) will be 
         * written to.
         *
         * Function that takes a request and a reference 
         * to an OutputBuffer.  Processes the query by invoking
         * the proper methods that constitute the QueryEngine
         * pipeline.  Returns a result set via OutputBuffer, or
         * if there is an error at any point during processing,
         * the proper error message.
         *
         * Invoked by TcpConnection with an incoming query
         * string
         *
         *
         * @see OutputBuffer
         * @see Parser
         */

        bool processRequest(const std::string &request, OutputBuffer &outputBuffer);


    private:
        std::string tcpPort_;
        int numThreads_;

        Metadata_Namespace::Catalog catalog_;

        /// The io_service used to perform asynchronous operations
        boost::asio::io_service ioService_;

        /// The signal_set is used to register for process termination notifications.
        boost::asio::signal_set signals_;

        TcpServer_Namespace::TcpServer tcpServer_;

        /**
         * @brief Registers a handler (stop()) for a set of
         * signals indicating database shutdown.
         *
         * Uses boost::asio::io_sersvice to register the stop()
         * handler on the SIGINT, SIGTERM, SIGQUIT signals.
         *
         * @see stop()
         */

        void registerSignals();
};

} // Database_Namespace


#endif

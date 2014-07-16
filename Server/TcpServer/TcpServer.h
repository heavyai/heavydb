#ifndef TCP_SERVER_H
#define TCP_SERVER_H

#include <string>
#include <vector>
#include <boost/asio.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "TcpConnection.h"

namespace Database_Namespace {
    class Database; //forward declaration of database class
}

using Database_Namespace::Database;

namespace TcpServer_Namespace {

/// The top-level class of the HTTP TcpServer.
class TcpServer : private boost::noncopyable
{
    public:
      /// Construct the TcpServer to listen on the specified TCP address and port, and
      /// serve up files from the given directory.
        explicit TcpServer(const std::string& address, const std::string& port, boost::asio::io_service &ioService, Database &database);
      /// Start listening for connections
      void start();

      /// Stop the TcpServer.
      void stop();

    private:
      /// Handle completion of an asynchronous accept operation.
        void handleAccept(const boost::system::error_code& e);

        void startAccept();

        /// The number of threads that will call io_service::run().
        std::size_t thread_pool_size_;

        /// The io_service used to perform asynchronous operations.
        boost::asio::io_service &ioService_;

        /// Reference to the database object
        Database &database_;

        /// Acceptor used to listen for incoming connections.
        boost::asio::ip::tcp::acceptor acceptor_;

        /// The next connection to be accepted.
        TcpConnection_ptr newConnection_;

        std::string address_;
        std::string port_;

};

} // namespace TcpServer_Namespace

#endif // TCP_SERVER_H

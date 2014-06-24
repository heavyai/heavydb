#ifndef TCP_SERVER_H
#define TCP_SERVER_H

#include <boost/asio.hpp>
#include <string>
#include <vector>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "TcpConnection.h"
#include "RequestHandler.h"

namespace TcpServer {

/// The top-level class of the HTTP TcpServer.
class TcpServer : private boost::noncopyable
{
    public:
      /// Construct the TcpServer to listen on the specified TCP address and port, and
      /// serve up files from the given directory.
      explicit TcpServer(const std::string& address, const std::string& port,
          std::size_t thread_pool_size);

      /// Run the TcpServer's io_service loop.
      void start();

      /// Stop the TcpServer.
      void stop();

    private:
      /// Handle completion of an asynchronous accept operation.
      void handle_accept(const boost::system::error_code& e);

      /// The number of threads that will call io_service::run().
      std::size_t thread_pool_size_;

      /// The io_service used to perform asynchronous operations.
      boost::asio::io_service io_service_;

      /// Acceptor used to listen for incoming connections.
      boost::asio::ip::tcp::acceptor acceptor_;

      /// The next connection to be accepted.
      TcpConnection_ptr new_connection_;

      /// The handler for all incoming requests.
      RequestHandler request_handler_;

      /// The signal_set is used to register for process termination notifications.
      boost::asio::signal_set signals_;
};

} // namespace TcpServer

#endif // TCP_SERVER_H

#ifndef TCP_CONNECTION_H
#define TCP_CONNECTION_H

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/logic/tribool.hpp>
#include "boost/tuple/tuple_comparison.hpp"
#include "Reply.h"
#include "Request.h"
#include "RequestHandler.h"
#include "RequestParser.h"

namespace TcpServer {

const int bufferMaxSize = 8192;

/// Represents a single TcpConnection from a client.
class TcpConnection : public boost::enable_shared_from_this<TcpConnection>, private boost::noncopyable 
{
    public:
      /// Construct a TcpConnection with the given io_service.
      explicit TcpConnection(boost::asio::io_service& io_service,
          RequestHandler& handler);

      /// Get the socket associated with the TcpConnection.
      boost::asio::ip::tcp::socket& socket();

      /// Start the first asynchronous operation for the TcpConnection.
      void start();

    private:
      /// Handle completion of a read operation.
      void handle_read(const boost::system::error_code& e,
          std::size_t bytes_transferred);

      /// Handle completion of a write operation.
      void handle_write(const boost::system::error_code& e);

      /// Strand to ensure the TcpConnection's handlers are not called concurrently.
      boost::asio::io_service::strand strand_;

      /// Socket for the connection.
      boost::asio::ip::tcp::socket socket_;

      /// The handler used to process the incoming request.
      RequestHandler& requestHandler_;

      /// Buffer for incoming data.
      boost::asio::streambuf buffer_;
      //boost::array<char, 8192> buffer_;

      /// The incoming request.
      request request_;

      /// The parser for the incoming request. - TODO
      //RequestParser request_parser_;

      /// The reply to be sent back to the client. - TODO
      //reply reply_;

      /// The query delimiter
      char queryDelim_;
};

typedef boost::shared_ptr<TcpConnection> TcpConnection_ptr;

} // namespace TcpServer

#endif // TCP_CONNECTION_H

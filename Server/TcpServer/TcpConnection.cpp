#include "TcpConnection.h"
#include <vector>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>

// DEBUG ONLY FIXME
#include <sstream>
#include <iostream>
#include <string>
//

namespace TcpServer {

TcpConnection::TcpConnection(boost::asio::io_service& io_service,
    RequestHandler& handler)
  : strand_(io_service),
    socket_(io_service),
    requestHandler_(handler),
    queryDelim_(';')
{
}

boost::asio::ip::tcp::socket& TcpConnection::socket()
{
  return socket_;
}

void TcpConnection::start()
{
  /*
  socket_.async_read_some(boost::asio::buffer(buffer_),
      strand_.wrap(
        boost::bind(&TcpConnection::handle_read, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred)));
  */
  boost::asio::async_read_until(socket_, buffer_,
      queryDelim_,
      /*strand_.wrap(*/
        boost::bind(&TcpConnection::handle_read, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred)); //);
}

void TcpConnection::handle_read(const boost::system::error_code& e,
    std::size_t bytes_transferred)
{
  if (!e)
  {

    bool isLegit;
    //std::ostringstream ss;
    const char* string = boost::asio::buffer_cast<const char*>(buffer_.data());
    std::string theReq(string, bytes_transferred);
    std::cout << "TcpConnection::handle_read() : " << theReq << std::endl;
    isLegit = requestHandler_.parse(request_, theReq); //.begin(), buffer_.begin() + bytes_transferred);

    // Request is good
    if (isLegit) {
      /*
      std::ostringstream ss;
      ss << boost::lexical_cast<std::string>(buffer_);
      std::string input = ss.str().substr(0,bytes_transferred);
      std::cout << "TcpConnection::handle_read() : " << input << std::endl;
      */
      /*
      request_handler_.handle_request(request_, reply_);
      boost::asio::async_write(socket_, reply_.to_buffers(),
          strand_.wrap(
            boost::bind(&TcpConnection::handle_write, shared_from_this(),
              boost::asio::placeholders::error)));
      */
    }
    // Bad request
    else if (0) //!result)
    {
      /*
      reply_ = reply::stock_reply(reply::bad_request);
      boost::asio::async_write(socket_, reply_.to_buffers(),
          strand_.wrap(
            boost::bind(&TcpConnection::handle_write, shared_from_this(),
              boost::asio::placeholders::error)));
      */
    }
    else
    {
      /*
      socket_.async_read_some(boost::asio::buffer(buffer_),
          strand_.wrap(
            boost::bind(&TcpConnection::handle_read, shared_from_this(),
              boost::asio::placeholders::error,
              boost::asio::placeholders::bytes_transferred)));
      */
    }
  }

  // If an error occurs then no new asynchronous operations are started. This
  // means that all shared_ptr references to the TcpConnection object will
  // disappear and the object will be destroyed automatically after this
  // handler returns. The TcpConnection class's destructor closes the socket.
}

void TcpConnection::handle_write(const boost::system::error_code& e)
{
  if (!e)
  {
    // Initiate graceful TcpConnection closure.
    boost::system::error_code ignored_ec;
    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ignored_ec);
  }

  // No new asynchronous operations are started. This means that all shared_ptr
  // references to the TcpConnection object will disappear and the object will be
  // destroyed automatically after this handler returns. The TcpConnection class's
  // destructor closes the socket.
}

} // namespace TcpServer

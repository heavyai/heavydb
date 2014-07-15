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

TcpConnection::~TcpConnection() {
    std::cout << "End of conn" << std::endl;
    boost::system::error_code ignored_ec;
    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ignored_ec);
    socket_.close();
}

boost::asio::ip::tcp::socket& TcpConnection::socket()
{
  return socket_;
}

void TcpConnection::start()
{
  printf("\nStarting connection\n");
  /*
  socket_.async_read_some(boost::asio::buffer(buffer_),
      strand_.wrap(
        boost::bind(&TcpConnection::handle_read, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred)));
  */
  boost::asio::async_read_until(socket_, buffer_,
      queryDelim_,
      strand_.wrap(
        boost::bind(&TcpConnection::handle_read, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred))); //);
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
    buffer_.consume(buffer_.size());
    std::cout << "TcpConnection::handle_read() : " << theReq << std::endl;
    isLegit = requestHandler_.parse(request_, theReq); //.begin(), buffer_.begin() + bytes_transferred);
    isLegit = true;

    // Request is good
    if (isLegit) {
        char data [24];
        int bytes = 16;
        long long int numRows = -1;
        memcpy(data, &bytes,sizeof(int));
        memcpy(data + 4, &numRows,sizeof(long long int));
        bytes = 4;
        memcpy(data + 12, &bytes,sizeof(int));
        data[16] = 'g';
        data[17] = 'o';
        data[18] = 'o';
        data[19] = 'd';
        bytes = 0;
        memcpy(data + 20, &bytes,sizeof(int));
        


        /*
      std::ostringstream ss;
      ss << boost::lexical_cast<std::string>(buffer_);
      std::string input = ss.str().substr(0,bytes_transferred);
      std::cout << "TcpConnection::handle_read() : " << input << std::endl;
      request_handler_.handle_request(request_, reply_);
      */
      boost::asio::async_write(socket_, boost::asio::buffer(data) /*reply_.to_buffers()*/,
          strand_.wrap(
            boost::bind(&TcpConnection::handle_write, shared_from_this(),
              boost::asio::placeholders::error)));
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

      boost::asio::async_read_until(socket_, buffer_,
          queryDelim_,
          strand_.wrap(
            boost::bind(&TcpConnection::handle_read, shared_from_this(),
              boost::asio::placeholders::error,
              boost::asio::placeholders::bytes_transferred))); //);

    // Initiate graceful TcpConnection closure.
      //std::cout << "Closing connection" << std::endl;
    //boost::system::error_code ignored_ec;
    //socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ignored_ec);
  }
  else {
      std::cout << "socket error: " << e << std::endl;
  }

  // No new asynchronous operations are started. This means that all shared_ptr
  // references to the TcpConnection object will disappear and the object will be
  // destroyed automatically after this handler returns. The TcpConnection class's
  // destructor closes the socket.
}

} // namespace TcpServer

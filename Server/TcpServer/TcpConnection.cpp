#include <vector>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>

#include "TcpConnection.h"
#include "../../Database/Database.h"
#include "../Output/OutputBuffer.h"

// DEBUG ONLY FIXME
#include <sstream>
#include <iostream>
#include <string>

using std::string;
using std::vector;
using std::cout;
using std::endl;

namespace TcpServer_Namespace {

TcpConnection::TcpConnection(boost::asio::io_service& ioService,
    Database& database)
  : strand_(ioService),
    socket_(ioService),
    database_(database),
    queryDelim_(';')
{
  printf("\nCreating connection\n");
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
        boost::bind(&TcpConnection::handleRead, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred))); //);
}

void TcpConnection::handleRead(const boost::system::error_code& e,
    std::size_t bytes_transferred)
{
    if (!e) {
        const char* string = boost::asio::buffer_cast<const char*>(buffer_.data());
        std::string request(string, bytes_transferred);
        buffer_.consume(buffer_.size());
        OutputBuffer outputBuffer;
        bool isValid = database_.processRequest(request, outputBuffer);
        writeOutput(outputBuffer);
    }
    // Bad request
    else { 
        std::cerr << "TcpServer - Error in handling read" << endl;
    }

  // If an error occurs then no new asynchronous operations are started. This
  // means that all shared_ptr references to the TcpConnection object will
  // disappear and the object will be destroyed automatically after this
  // handler returns. The TcpConnection class's destructor closes the socket.
}

void TcpConnection::writeOutput(OutputBuffer &outputBuffer) {
    if (outputBuffer.size() > 0) {
        const vector <char> subBuffer = outputBuffer.front();
        boost::asio::const_buffer writeBuffer = boost::asio::buffer(subBuffer,subBuffer.size());
        outputBuffer.pop();

        boost::asio::async_write(socket_, /*boostBuffer, */ boost::asio::buffer(writeBuffer), 
          strand_.wrap(
            boost::bind(&TcpConnection::handleWrite, shared_from_this(),
              boost::asio::placeholders::error, outputBuffer)));

    }
    else {
        //cout << "New cycle" << endl;
        start();
    }
}



void TcpConnection::handleWrite(const boost::system::error_code& e, OutputBuffer &outputBuffer)
{
  if (!e)
  {
      writeOutput(outputBuffer);
  }
  else {
      std::cout << "socket error: " << e << std::endl;
  }

  // No new asynchronous operations are started. This means that all shared_ptr
  // references to the TcpConnection object will disappear and the object will be
  // destroyed automatically after this handler returns. The TcpConnection class's
  // destructor closes the socket.
}

} // namespace TcpServer_Namespace

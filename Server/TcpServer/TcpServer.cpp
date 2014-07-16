#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include "TcpServer.h"
#include "TcpConnection.h"
#include "RequestHandler.h"
#include "../../Database/Database.h"

namespace TcpServer_Namespace {

TcpServer::TcpServer(const std::string& address, const std::string& port, boost::asio::io_service &ioService, Database &database)
  : address_(address),
    port_(port),
    ioService_(ioService),
    acceptor_(ioService_),
    database_(database),
    newConnection_()
{}

void TcpServer::start()
{
  // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
  boost::asio::ip::tcp::resolver resolver(ioService_);
  boost::asio::ip::tcp::resolver::query query(address_, port_);
  boost::asio::ip::tcp::endpoint endpoint = *resolver.resolve(query);
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();
  startAccept();
}

void TcpServer::stop()
{
  printf("\nTcpServer stopping()\n");
  acceptor_.close();
  //io_service_.stop();
}

void TcpServer::startAccept() {
      newConnection_.reset(new TcpConnection(ioService_,database_));
      acceptor_.async_accept(newConnection_ ->socket(),
      boost::bind(&TcpServer::handleAccept, this, 
        boost::asio::placeholders::error));
}

void TcpServer::handleAccept(const boost::system::error_code& e)
{
  if (!e) {
    newConnection_->start();
  }
  startAccept();
}


} // namespace TcpServer_Namespace

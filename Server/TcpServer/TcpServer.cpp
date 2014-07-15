#include "TcpServer.h"
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include "TcpConnection.h"
#include "RequestHandler.h"

namespace TcpServer {

TcpServer::TcpServer(const std::string& address, const std::string& port,
    std::size_t thread_pool_size/*, Database *database*/)
  : thread_pool_size_(thread_pool_size),
    signals_(io_service_),
    acceptor_(io_service_),
    newConnection_()
{
  // Register to handle the signals that indicate when the server should exit.
  // It is safe to register for the same signal multiple times in a program,
  // provided all registration for the specified signal is made through Asio.
  signals_.add(SIGINT);
  signals_.add(SIGTERM);
#if defined(SIGQUIT)
  signals_.add(SIGQUIT);
#endif // defined(SIGQUIT)
  signals_.async_wait(boost::bind(&TcpServer::stop, this));

  // Open the acceptor with the option to reuse the address (i.e. SO_REUSEADDR).
  boost::asio::ip::tcp::resolver resolver(io_service_);
  boost::asio::ip::tcp::resolver::query query(address, port);
  boost::asio::ip::tcp::endpoint endpoint = *resolver.resolve(query);
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();
  startAccept();

}

void TcpServer::start()
{
  // Create a pool of threads to run all of the io_services.
  std::vector<boost::shared_ptr<boost::thread> > threads;
  for (std::size_t i = 0; i < thread_pool_size_; ++i)
  {
    boost::shared_ptr<boost::thread> thread(new boost::thread(
          boost::bind(&boost::asio::io_service::run, &io_service_)));
    threads.push_back(thread);
  }

  // Wait for all threads in the pool to exit.
  for (std::size_t i = 0; i < threads.size(); ++i)
    threads[i]->join();
}

void TcpServer::stop()
{
  printf("\nCaught a kill signal, calling stop()\n");
  acceptor_.close();
  io_service_.stop();
}

void TcpServer::startAccept() {
      newConnection_.reset(new TcpConnection(io_service_,request_handler_));
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


} // namespace TcpServer

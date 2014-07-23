#include <iostream>
#include "Database.h"
#include "OutputBuffer.h"
#include "OutputWriter.h"
#include "parser.h"


using namespace std;

namespace Database_Namespace {

Database::Database(const std::string &tcpPort, const int numThreads): tcpPort_(tcpPort), numThreads_(numThreads), signals_(ioService_),  tcpServer_("0.0.0.0",tcpPort,ioService_, *this)   { 
    registerSignals();
}

void Database::registerSignals() {
  // Register to handle the signals that indicate when the server should exit.
  // It is safe to register for the same signal multiple times in a program,
  // provided all registration for the specified signal is made through Asio.
  cout << "Registering signals" << endl;
  signals_.add(SIGINT);
  signals_.add(SIGTERM);
#if defined(SIGQUIT)
  signals_.add(SIGQUIT);
#endif // defined(SIGQUIT)
  signals_.async_wait(boost::bind(&Database::stop, this));
}

void Database::start() {
  // Create a pool of threads to run all of the io_services.
  std::vector<boost::shared_ptr<boost::thread> > threads;
  for (std::size_t i = 0; i < numThreads_; ++i)
  {
    boost::shared_ptr<boost::thread> thread(new boost::thread(
          boost::bind(&boost::asio::io_service::run, &ioService_)));
    threads.push_back(thread);
  }

  tcpServer_.start();

  // Wait for all threads in the pool to exit.
  // use join_all?
  for (std::size_t i = 0; i < threads.size(); ++i)
    threads[i]->join();
}

void Database::stop() {
    std::cout << "Caught a kill signal - database stopping" << std::endl;
    tcpServer_.stop();

    // would need to checkpoint here

    ioService_.stop();
}

bool Database::processRequest(const std::string &request, OutputBuffer &outputBuffer) {
    std::cout << "Request: " << request << std::endl;
    OutputWriter outputWriter(outputBuffer);
    Parser parser;
    ASTNode *parseRoot = 0;
    string lastParsed;
    int numErrors = parser.parse(request, parseRoot,lastParsed);
    if (numErrors > 0) {
        string errorString ("Parsing error at " + lastParsed );
        outputWriter.writeError(errorString);
        return false;
    }
    else {
        outputWriter.writeStatusMessage("OK");
        return true;
    }
}
        
} // Database_Namespace



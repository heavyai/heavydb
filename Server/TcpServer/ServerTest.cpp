#include <boost/lexical_cast.hpp>
#include <iostream>
#include "TcpServer.h"

int main(int argc, char* argv[])
{
    try
    {
        // Check command line arguments.
        if (argc != 4)
        {
          std::cerr << "Usage: tcpServer <address> <port> <threads> \n";
          std::cerr << "  For IPv4, try:\n";
          std::cerr << "    http_server 0.0.0.0 80 1 .\n";
          std::cerr << "  For IPv6, try:\n";
          std::cerr << "    http_server 0::0 80 1 .\n";
          return 1;
        }

        // Initialize server.
        std::size_t numThreads = boost::lexical_cast<std::size_t>(argv[3]);
        TcpServer::TcpServer testServer(argv[1], argv[2], numThreads);

        // Run the server until stopped.
        testServer.start();
    }
    catch (std::exception& e)
    {
        std::cerr << "exception: " << e.what() << "\n";
    }

    return 0;

}

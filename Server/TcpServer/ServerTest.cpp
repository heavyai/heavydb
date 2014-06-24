#include "TcpServer.h"

#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <signal.h>
#include <unistd.h>


void sigHandler(int sigNumber) {
    // Ctrl-C sends this
    if (sigNumber == SIGINT) {
        printf("\nReceived SIGINT\n");
        exit(0);
    }
    // lsb_init_functions sends this first
    else if (sigNumber == SIGTERM) {
        printf("\nReceived SIGTERM\n");
        exit(0);
    }
    else {
        printf("\nSignal not recognized\n");
        exit(1);        
    }
}

void registerSignals(void) {
    if (signal(SIGINT, sigHandler) == SIG_ERR) {
        printf("\nCan't catch SIGINT\n");
    }
    if (signal(SIGTERM, sigHandler) == SIG_ERR) {
        printf("\nCan't catch SIGTERM\n");
    }
}

int main(int argc, char* argv[])
{
    registerSignals();

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
        std::size_t num_threads = boost::lexical_cast<std::size_t>(argv[3]);
        Server::TcpServer s(argv[1], argv[2], num_threads);

        // Run the server until stopped.
        s.start();
    }
    catch (std::exception& e)
    {
        std::cerr << "exception: " << e.what() << "\n";
    }

    return 0;

}

#ifndef REPLY_H
#define REPLY_H

#include <string>

namespace TcpServer {

/// A request received from a client.
struct reply
{
  std::string response;
  int server_version_major;
  int server_version_minor;
};

} // namespace TcpServer

#endif // REPLY_H

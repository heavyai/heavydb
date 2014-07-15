#ifndef REQUEST_H
#define REQUEST_H

#include <string>

namespace TcpServer {

/// A request received from a client.
struct request
{
  std::string query;
  int client_version_major;
  int client_version_minor;
};

} // namespace TcpServer

#endif // REQUEST_H

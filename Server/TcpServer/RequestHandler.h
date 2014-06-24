#ifndef REQUEST_HANDLER_H
#define REQUEST_HANDLER_H

#include <string>
#include <boost/noncopyable.hpp>
//#include "Reply.h"
#include "Request.h"

namespace TcpServer {

/// The common handler for all incoming requests.
class RequestHandler : private boost::noncopyable
{
    public:
      /// Construct with a parser object most likely
      explicit RequestHandler() {};

      /// Handle a request and produce a reply.
      void handle_request(const request& req /*, reply& rep*/ );

    private:

      /// Parse the request to ensure validity - call the parser's main function here
      static bool parse(const request& theRequest);
};

} // namespace TcpServer

#endif // REQUEST_HANDLER_H

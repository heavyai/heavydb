#include "RequestHandler.h"
#include <fstream>
#include <sstream>
#include <string>
#include <boost/lexical_cast.hpp>
//#include "Reply.h"
//#include "Request.h"

namespace TcpServer {

/*
RequestHandler::RequestHandler()
{
    int x = 1;
}
*/

void RequestHandler::handle_request(const request& theRequest /*, reply& theReply*/ )
{
  if (!parse(theRequest))
  {
    //theReply = reply::stock_reply(reply::bad_request);
    return;
  }

  // Fill out the reply to be sent to the client.
  /*
  rep.status = reply::ok;
  char buf[512];
  while (is.read(buf, sizeof(buf)).gcount() > 0)
    rep.content.append(buf, is.gcount());
  rep.headers.resize(2);
  rep.headers[0].name = "Content-Length";
  rep.headers[0].value = boost::lexical_cast<std::string>(rep.content.size());
  rep.headers[1].name = "Content-Type";
  rep.headers[1].value = mime_types::extension_to_type(extension);
  */
}

bool RequestHandler::parse(const request& theReqest, InputIterator begin, InputIterator end)
{
    while (begin != end) {
        /*
        boost::tribool result = consume(req, *begin++);
        if (result || !result) {
            return boost::make_tuple(result, begin);
        }
        */
    }
    // TODO
    //return true;
}

} // namespace TcpServer

#ifndef TCP_CONNECTION_H
#define TCP_CONNECTION_H

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace Database_Namespace {
    class Database; //forward declaration of database class
}

class OutputBuffer; //forward declaration

using Database_Namespace::Database;

namespace TcpServer_Namespace {

const int bufferMaxSize = 8192;

/**
 * @type TcpConnection
 * @brief represents a single tcp connection from a client
 *
 * Can be kept open over multiple queries.
 */
class TcpConnection : public boost::enable_shared_from_this<TcpConnection>, private boost::noncopyable 
{
    public:
      /**
       * @brief Construct a TcpConnection with an already extant
       * io_service and a reference to Database
       * @param ioService already instanciated boost::asio::io_service
       * @param database Database object to pass queries to
       *
       * @see Database
       */

      explicit TcpConnection(boost::asio::io_service& ioService,
          Database &database);
    ~TcpConnection();

      /**
       * @brief Get the socket associated with the TcpConnection.
       */

      boost::asio::ip::tcp::socket& socket();

      /*
       * @brief Triggers TcpConnection to wait for next input (read)
       */
      void start();

    private:
        /*
         * @brief Writes one SubBuffer of 
         * output from OutputBuffer queue
         * @param outputBuffer queue of data transmissions to
         * be sent to client
         *
         * Called by handleRead after query is 
         * processed and then by handleWrite 
         * repeatedly until OutputBuffer queue
         * is empty.
         *
         * @see OutputBuffer
         */

        void writeOutput(OutputBuffer &outputBuffer);


        /*
         * @brief Handle completion of a read operation.
         * @param e error code
         * @bytes_transferred size of message
         * received in bytes
         *
         * At this point query should have been 
         * fully received.  This method creates
         * an OutputBuffer and then sends the 
         * query and sends the received query
         * and the OutputBuffer to 
         * Database::processRequest. After
         * response is received calls 
         * writeOutput.
         
         * @see Database_Namespace::Database::processRequest
         * @see writeOutput
         * 
         */

        void handleRead(const boost::system::error_code& e,
          std::size_t bytes_transferred);

        /*
         * @brief Handle completion of a write operation.
         * @param e error code
         * @param outputBuffer reference to OutputBuffer
         *
         * If no error then calls writeOutput repeatedly
         *
         * @see writeOutput()
         */

        void handleWrite(const boost::system::error_code& e, OutputBuffer &outputBuffer);

        /// Strand to ensure the TcpConnection's handlers are not called concurrently.
        boost::asio::io_service::strand strand_;

        /// Socket for the connection.
        boost::asio::ip::tcp::socket socket_;

        /// The handler used to process the incoming request.
        //RequestHandler& requestHandler_;
        Database & database_;



/// Buffer for incoming data.
boost::asio::streambuf buffer_;
      //boost::array<char, 8192> buffer_;

      /// The incoming request.
      //request request_;

      /// The parser for the incoming request. - TODO
      //RequestParser request_parser_;

      /// The reply to be sent back to the client. - TODO
      //reply reply_;

      /// The query delimiter
      char queryDelim_;
};

typedef boost::shared_ptr<TcpConnection> TcpConnection_ptr;

} // namespace TcpServer_Namespace

#endif // TCP_CONNECTION_H

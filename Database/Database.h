#ifdef DATABASE_H
#define DATABASE_H

class Database {


    public:
        Database(const int tcpPort, const int numThreads); 


    private:
        int tcpPort_;
        int numThreads_;

        boost::asio::io_service ioService_;
        TcpServer tcpServer_;







}


#endif

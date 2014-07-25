class PgConnector {
    public:
        PGConnector (const std::string &dbName, const std::string &userName, const std::string &host = "localhost", const std::string &port = "5432") 


    private:
        std::string dbName_;
        std::string userName_;
        std::string port_;
        std::string host_;
        std::string connectString_;





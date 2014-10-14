#include <iostream>
#include "../Shared/types.h"
#include "Database.h"
#include "OutputBuffer.h"
#include "OutputWriter.h"
#include "../QueryEngine/Parse/SQL/parser.h"
#include "../QueryEngine/Parse/SQL/visitor/XMLTranslator.h"
#include "../QueryEngine/Parse/RA/visitor/XMLTranslator.h"
#include "../QueryEngine/Plan/Planner.h"
#include "../QueryEngine/Plan/Translator.h"
//#include "FileMgr.h"
//#include "BufferMgr.h"
#include "../DataMgr/Metadata/Catalog.h"
#include "../DataMgr/Partitioner/TablePartitionMgr.h"


#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>


using namespace std;
using namespace Plan_Namespace;

namespace Database_Namespace {

    Database::Database(const std::string &tcpPort, const int numThreads) :
    tcpPort_(tcpPort),
    numThreads_(numThreads),
    signals_(ioService_),
    tcpServer_("0.0.0.0",tcpPort,ioService_, *this)
    {
        registerSignals();
        //fileMgr_ = new File_Namespace::FileMgr ("data");
        //bufferMgr_ = new Buffer_Namespace::BufferMgr (128*1048576, fileMgr_);
        catalog_ = new Metadata_Namespace::Catalog ("data");
        //tablePartitionMgr_ = new Partitioner_Namespace::TablePartitionMgr (*catalog_, *bufferMgr_);
    }
    
    Database::~Database() {
        // delete tablePartitionMgr_;
        delete catalog_;
        // delete bufferMgr_;
        // delete fileMgr_;
    }
    
    bool Database::processRequest(const std::string &request, OutputBuffer &outputBuffer, bool printToStdout) {
        
        // Parse request (an SQL string)
        SQLParser parser;
        ASTNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(request, parseRoot, lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            return false;
        }
        else {
            if (printToStdout) {
                cout << "\tSQL ABSTRACT SYNTAX TREE:" << endl;
                SQL_Namespace::XMLTranslator sql2xml;
                parseRoot->accept(sql2xml);
                cout << endl;
            }
        }
        
        // Translate SQL AST to a query plan
        Translator sql2plan(*catalog_);
        Planner planner(sql2plan);
        QueryStmtType stmtType;
        
        AbstractPlan *plan = planner.makePlan(request, stmtType);
        
        if (stmtType == QUERY_STMT) {
            RA_Namespace::RelAlgNode *qp = (RelAlgNode*)plan->getPlan();
            if (printToStdout) {
                cout << "\tQUERY PLAN:" << endl;
                RA_Namespace::XMLTranslator ra2qp;
                qp->accept(ra2qp);
            }
        }
        
        return true;
    }

    /**
     * Register to handle the signals that indicate when the server should exit.
     * It is safe to register for the same signal multiple times in a program,
     * provided all registration for the specified signal is made through Asio.
     */
    void Database::registerSignals() {
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
        for (std::size_t i = 0; i < numThreads_; ++i) {
            boost::shared_ptr<boost::thread> thread(new boost::thread(boost::bind(&boost::asio::io_service::run, &ioService_)));
            threads.push_back(thread);
        }
        tcpServer_.start();
        
        // Wait for all threads in the pool to exit.
        // @todo use join_all?
        for (std::size_t i = 0; i < threads.size(); ++i)
            threads[i]->join();
    }
    
    void Database::stop() {
        std::cout << "Caught a kill signal - database stopping" << std::endl;
        tcpServer_.stop();
        // @todo would need to checkpoint here
        ioService_.stop();
    }
        
} // Database_Namespace



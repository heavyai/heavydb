#ifndef MT_SQL_PARSER_H
#define	MT_SQL_PARSER_H

#include "parser.h"

#include <mutex>

class MTSQLParser {
public:
  int parse(const std::string& inputStr, std::list<Stmt*>& parseTrees, std::string& lastParsed) {
    std::lock_guard<std::mutex> lock(mutex_);
    return parser_.parse(inputStr, parseTrees, lastParsed);
  }
private:
  SQLParser parser_;
  std::mutex mutex_;
};

#endif	// MT_SQL_PARSER_H


#ifndef QUERYENGINE_RELALGABSTRACTINTERPRETER_H
#define QUERYENGINE_RELALGABSTRACTINTERPRETER_H

#include <rapidjson/document.h>

class ScanScope {
  // TODO
};

class ScanBufferDesc {
 public:
  ScanBufferDesc();                    // for results of other queries
  ScanBufferDesc(const int table_id);  // for tables

 private:
  const int table_id_;
};

class LoweringInfo {
  // TODO
};

LoweringInfo ra_interpret(const rapidjson::Value&);

#endif  // QUERYENGINE_RELALGABSTRACTINTERPRETER_H

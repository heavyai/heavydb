#ifndef QUERYENGINE_RELALGABSTRACTINTERPRETER_H
#define QUERYENGINE_RELALGABSTRACTINTERPRETER_H

#include "../Catalog/Catalog.h"

#include <rapidjson/document.h>

class ScanScope {
  // TODO
};

class ScanBufferDesc {
 public:
  ScanBufferDesc();                           // for results of other queries
  ScanBufferDesc(const TableDescriptor* td);  // for tables

 private:
  const TableDescriptor* td_;
};

class LoweringInfo {
  // TODO
};

LoweringInfo ra_interpret(const rapidjson::Value&, const Catalog_Namespace::Catalog&);

#endif  // QUERYENGINE_RELALGABSTRACTINTERPRETER_H

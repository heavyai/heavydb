#include "RelAlgAbstractInterpreter.h"

#include <glog/logging.h>

ScanBufferDesc::ScanBufferDesc() : table_id_(-1) {
}

ScanBufferDesc::ScanBufferDesc(const int table_id) : table_id_(table_id) {
  CHECK(table_id_ >= 0);
}

LoweringInfo ra_interpret(const rapidjson::Value&) {
  CHECK(false);
}

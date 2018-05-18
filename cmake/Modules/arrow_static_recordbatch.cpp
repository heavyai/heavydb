#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"

int main(int argc, char** argv) {
  std::vector<std::shared_ptr<arrow::Array>> result_columns;
  std::shared_ptr<arrow::Schema> schema;
  auto record = arrow::RecordBatch::Make(schema, 0, result_columns);
  // auto record = std::make_shared<arrow::RecordBatch>(schema, 0, result_columns);
}

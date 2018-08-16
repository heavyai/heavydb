#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/api.h"

int main(int argc, char** argv) {
  arrow::BooleanBuilder builder;
  std::vector<bool> zeros(1);
  builder.AppendValues(zeros);
}

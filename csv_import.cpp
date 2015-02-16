#include "Import/CsvImport.h"

#include <glog/logging.h>


int main(int argc, char** argv) {
  CHECK_GT(argc, 2);
  CsvImporter importer(argv[1], argv[2]);
  importer.import();
  return 0;
}

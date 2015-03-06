#include "Import/CsvImport.h"

#include <iostream>
#include "Shared/measure.h"
#include <glog/logging.h>


int main(int argc, char** argv) {
  CHECK_GT(argc, 2);
  CsvImporter importer(argv[1], argv[2]);
  auto ms = measure<>::execution([&]() {
  importer.import(); });
  std::cout << "Total Import Time: " << (double)ms/1000.0 << " Seconds." << std::endl;
  return 0;
}

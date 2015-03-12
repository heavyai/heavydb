#include "Import/CsvImport.h"

#include <iostream>
#include "Shared/measure.h"
#include <glog/logging.h>


int main(int argc, char** argv) {
  CHECK_GE(argc, 3);
  CsvImporter importer(argv[1], argc >= 4 ? argv[3] : "/tmp", argv[2]);
  auto ms = measure<>::execution([&]() {
  importer.import(); });
  std::cout << "Total Import Time: " << (double)ms/1000.0 << " Seconds." << std::endl;
  return 0;
}

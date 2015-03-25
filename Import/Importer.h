/*
 * @file Importer.h
 * @author Wei Hong < wei@mapd.com>
 * @brief Importer class for table import from file
 */
#ifndef _IMPORTER_H_
#define _IMPORTER_H_

#include <string>
#include <cstdio>
#include <cstdlib>
#include "../Catalog/TableDescriptor.h"
#include "../Catalog/Catalog.h"

struct CopyParams {
  char delimiter;
  std::string null_str;
  bool has_header;
  char quote;
  char escape;

  CopyParams() : delimiter(','), null_str("\\N"), has_header(true), quote('"'), escape('"') {}
};

#define IMPORT_FILE_BUFFER_SIZE   100000000

class Importer {
  public:
    Importer(const Catalog_Namespace::Catalog &c, const TableDescriptor *t, const std::string &f, const CopyParams &p);
    ~Importer();
    void import();
  private:
    const Catalog_Namespace::Catalog &catalog;
    const TableDescriptor *table_desc;
    const std::string &file_path;
    const CopyParams &copy_params;
    size_t file_size;
    int max_threads;
    FILE *p_file;
    char *buffer[2];
    int which_buf;
    std::list <const ColumnDescriptor *> column_descs;
};

#endif // _IMPORTER_H_

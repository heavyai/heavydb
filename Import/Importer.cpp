/*
 * @file Importer.cpp
 * @author Wei Hong <wei@mapd.com>
 * @brief Functions for Importer class
 */

#include <cstdio>
#include <unistd.h>
#include "Importer.h"

Importer::Importer(const Catalog_Namespace::Catalog &c, const TableDescriptor *t, const std::string &f, const CopyParams &p) : catalog(c), table_desc(t), file_path(f), copy_params(p) 
{
  file_size = 0;
  max_threads = 0;
  p_file = nullptr;
  buffer[0] = nullptr;
  buffer[1] = nullptr;
  which_buf = 0;
}

Importer::~Importer()
{
  if (p_file != nullptr)
    fclose(p_file);
  if (buffer[0] != nullptr)
    free(buffer[0]);
  if (buffer[1] != nullptr)
    free(buffer[1]);
}

void
Importer::import()
{
  column_descs = catalog.getAllColumnMetadataForTable(table_desc->tableId);
  p_file = fopen(file_path.c_str(), "r");
  (void)fseek(p_file,0,SEEK_END);
  file_size = ftell(p_file);
  max_threads = sysconf(_SC_NPROCESSORS_CONF);
  buffer[0] = (char*)malloc(IMPORT_FILE_BUFFER_SIZE);
  buffer[1] = (char*)malloc(IMPORT_FILE_BUFFER_SIZE);
  if (copy_params.escape == copy_params.quote)
    max_threads = 1;
  while (true) {
    size_t size = fread((void*)buffer[which_buf], 1, IMPORT_FILE_BUFFER_SIZE, p_file);
    if (size < IMPORT_FILE_BUFFER_SIZE && feof(p_file))
      break;

  }

  fclose(p_file); 
}

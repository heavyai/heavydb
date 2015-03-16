#ifndef IMPORT_CSVIMPORT_H
#define IMPORT_CSVIMPORT_H

#include "./csv_parser.hpp"

#include "../Catalog/Catalog.h"

#include <list>
#include <memory>
#include <string>


class MapDMeta {
public:
  MapDMeta(const std::string table_name, const std::string& base_data_path);
  const std::list<const ColumnDescriptor*>& getColumnDescriptors() const;
  int getTableId() const;
  const TableDescriptor* getTableDesc() const;
  int getDbId() const;
  Data_Namespace::DataMgr* getDataMgr() const;
  std::string getStringDictFolder(const int col_id) const;
  std::string getStringDictFolder(
    const int db_id,
    const int table_id,
    const int col_id);
  static std::string getStringDictFolder(
    const std::string& base_data_path,
    const int db_id,
    const int table_id,
    const int col_id);
private:
  Catalog_Namespace::Catalog* cat_;
  std::unique_ptr<Data_Namespace::DataMgr> data_mgr_;

  const std::string table_name_;
  const TableDescriptor* td_;
  int table_id_;
  std::list<const ColumnDescriptor*> col_descriptors_;
  const std::string db_name_ { MAPD_SYSTEM_DB };
  const std::string user_ { MAPD_ROOT_USER };
  const std::string pass_ { MAPD_ROOT_PASSWD_DEFAULT };
  const std::string base_data_path_;
};

class CsvImporter {
public:
  CsvImporter(
    const std::string& table_name,
    const std::string& base_data_path,
    const std::string& file_path,
    const std::string& delim = ",");
  void import();
  ~CsvImporter();
private:
  const std::string table_name_;
  const std::string file_path_;
  MapDMeta table_meta_;
  csv_parser csv_parser_;
};

#endif  // IMPORT_CSVIMPORT_H

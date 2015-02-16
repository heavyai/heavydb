#include "CsvImport.h"
#include "csvparser.h"

#include <boost/filesystem.hpp>
#include <glog/logging.h>

#include <cstdio>
#include <list>


MapDMeta::MapDMeta(const std::string table_name)
  : table_name_(table_name)
  , table_id_(-1) {
  CHECK(boost::filesystem::exists(base_path_));
  const auto system_db_file = boost::filesystem::path(base_path_) / "mapd_catalogs" / "mapd";
  CHECK(boost::filesystem::exists(system_db_file));
  const auto data_path = boost::filesystem::path(base_path_) / "mapd_data";
  data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string()));
  Catalog_Namespace::SysCatalog sys_cat(base_path_, *data_mgr_);
  Catalog_Namespace::UserMetadata user_meta;
  CHECK(sys_cat.getMetadataForUser(user_, user_meta));
  CHECK_EQ(user_meta.passwd, pass_);
  Catalog_Namespace::DBMetadata db_meta;
  CHECK(sys_cat.getMetadataForDB(db_name_, db_meta));
  CHECK(user_meta.isSuper || user_meta.userId == db_meta.dbOwner);
  cat_.reset(new Catalog_Namespace::Catalog(base_path_, user_meta, db_meta, *data_mgr_));
  td_ = cat_->getMetadataForTable(table_name_);
  CHECK(td_);
  table_id_ = td_->tableId;
  col_descriptors_ = cat_->getAllColumnMetadataForTable(table_id_);
}

const std::list<const ColumnDescriptor*>& MapDMeta::getColumnDescriptors() const {
  return col_descriptors_;
}

int MapDMeta::getTableId() {
  return -1;
}

const TableDescriptor* MapDMeta::getTableDesc() const {
  return td_;
}

int MapDMeta::getDbId() {
  return cat_->get_currentDB().dbId;
}

Data_Namespace::DataMgr* MapDMeta::getDataMgr() const {
  auto& dm = cat_->get_dataMgr();
  return &dm;
}

namespace {

class TypedImportBuffer {
public:
  TypedImportBuffer(const SQLTypes type) : type_(type) {
    switch (type) {
    case kSMALLINT:
      smallint_buffer_ = new std::vector<int16_t>();
      break;
    case kINT:
      int_buffer_ = new std::vector<int32_t>();
      break;
    case kBIGINT:
      bigint_buffer_ = new std::vector<int64_t>();
      break;
    default:
      CHECK(false);
    }
  }

  ~TypedImportBuffer() {
    switch (type_) {
    case kSMALLINT:
      delete smallint_buffer_;
      break;
    case kINT:
      delete int_buffer_;
      break;
    case kBIGINT:
      delete bigint_buffer_;
      break;
    default:
      CHECK(false);
    }
  }

  void addSmallint(const int16_t v) {
    CHECK_EQ(kSMALLINT, type_);
    smallint_buffer_->push_back(v);
  }

  void addInt(const int32_t v) {
    CHECK_EQ(kINT, type_);
    int_buffer_->push_back(v);
  }

  void addBigint(const int64_t v) {
    CHECK_EQ(kBIGINT, type_);
    bigint_buffer_->push_back(v);
  }

  SQLTypes getType() const {
    return type_;
  }

  int8_t* getBytes() const {
    switch (type_) {
    case kSMALLINT:
      return reinterpret_cast<int8_t*>(&((*smallint_buffer_)[0]));
    case kINT:
      return reinterpret_cast<int8_t*>(&((*int_buffer_)[0]));
    case kBIGINT:
      return reinterpret_cast<int8_t*>(&((*bigint_buffer_)[0]));
    default:
      CHECK(false);
    }
  }
private:
  union {
    std::vector<int16_t>* smallint_buffer_;
    std::vector<int32_t>* int_buffer_;
    std::vector<int64_t>* bigint_buffer_;
  };
  SQLTypes type_;
};

};

CsvImporter::CsvImporter(
    const std::string& table_name,
    const std::string& file_path,
    const std::string& delim,
    const bool has_header)
  : table_name_(table_name)
  , table_meta_(table_name)
  , has_header_(has_header)
  , csv_parser_(CsvParser_new(file_path.c_str(), delim.c_str(), has_header)) {}

void CsvImporter::import() {
  const size_t row_buffer_size { 50000 };
  const auto col_descriptors = table_meta_.getColumnDescriptors();
  if (has_header_) {
    auto header = CsvParser_getHeader(csv_parser_);
    CHECK(header);
    CHECK_EQ(CsvParser_getNumFields(header), col_descriptors.size());
    char **header_fields = CsvParser_getFields(header);
    int col_idx = 0;
    for (const auto col_desc : col_descriptors) {
      CHECK_EQ(col_desc->columnName, header_fields[col_idx]);
      ++col_idx;
    }
    CsvParser_destroy_row(header);
  }
  std::vector<TypedImportBuffer> import_buffers;
  for (const auto col_desc : col_descriptors) {
    import_buffers.emplace_back(col_desc->columnType.type);
  }
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = table_meta_.getDbId();
  insert_data.tableId = table_meta_.getTableId();
  for (const auto col_desc : col_descriptors) {
    insert_data.columnIds.push_back(col_desc->columnId);
  }
  size_t row_count = 0;
  while (auto row = CsvParser_getRow(csv_parser_)) {
    char **row_fields = CsvParser_getFields(row);
    CHECK_EQ(CsvParser_getNumFields(row), col_descriptors.size());
    int col_idx = 0;
    for (const auto col_desc : col_descriptors) {
    switch (col_desc->columnType.type) {
    case kSMALLINT:
      import_buffers[col_idx].addSmallint(boost::lexical_cast<int16_t>(row_fields[col_idx]));
      break;
    case kINT:
      import_buffers[col_idx].addInt(boost::lexical_cast<int32_t>(row_fields[col_idx]));
      break;
    case kBIGINT:
      import_buffers[col_idx].addBigint(boost::lexical_cast<int64_t>(row_fields[col_idx]));
      break;
    default:
      CHECK(false);
    }
    ++col_idx;
    }
    CsvParser_destroy_row(row);
    ++row_count;
    if (row_count == row_buffer_size) {
      {
        decltype(insert_data.data) empty;
        insert_data.data.swap(empty);
      }
      insert_data.numRows = row_buffer_size;
      for (const auto& import_buff : import_buffers) {
        DataBlockPtr p;
        if (IS_INTEGER(import_buff.getType())) {
          p.numbersPtr = import_buff.getBytes();
          insert_data.data.push_back(p);
        } else {
          CHECK(false);
        }
      }
      table_meta_.getTableDesc()->fragmenter->insertData(insert_data);
      table_meta_.getDataMgr()->checkpoint();
      row_count = 0;
      {
        decltype(import_buffers) empty;
        import_buffers.swap(empty);
      }
    }
  }
}

CsvImporter::~CsvImporter() {
  CsvParser_destroy(csv_parser_);
}

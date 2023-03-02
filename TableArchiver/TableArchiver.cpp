/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "TableArchiver/TableArchiver.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/range/combine.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/version.hpp>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <list>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <system_error>

#include "Archive/S3Archive.h"
#include "DataMgr/FileMgr/FileInfo.h"
#include "DataMgr/FileMgr/FileMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"
#include "Parser/ParserNode.h"
#include "Shared/File.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Shared/ThreadController.h"
#include "Shared/file_path_util.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/thread_count.h"

extern bool g_cluster;
extern std::string g_base_path;
bool g_test_rollback_dump_restore{false};

constexpr static int kDumpVersion = 1;
constexpr static int kDumpVersion_remove_render_group_columns = 1;

constexpr static char const* table_schema_filename = "_table.sql";
constexpr static char const* table_oldinfo_filename = "_table.oldinfo";
constexpr static char const* table_epoch_filename = "_table.epoch";
constexpr static char const* table_dumpversion_filename = "_table.dumpversion";

#if BOOST_VERSION < 107300
namespace std {

template <typename T, typename U>
struct tuple_size<boost::tuples::cons<T, U>>
    : boost::tuples::length<boost::tuples::cons<T, U>> {};
template <size_t I, typename T, typename U>
struct tuple_element<I, boost::tuples::cons<T, U>>
    : boost::tuples::element<I, boost::tuples::cons<T, U>> {};

}  // namespace std
#endif

namespace {

inline auto simple_file_closer = [](FILE* f) { std::fclose(f); };

inline std::string abs_path(const File_Namespace::GlobalFileMgr* global_file_mgr) {
  return boost::filesystem::canonical(global_file_mgr->getBasePath()).string();
}

inline std::string run(const std::string& cmd,
                       const std::string& chdir = "",
                       const bool log_failure = true) {
  VLOG(3) << "running cmd: " << cmd;
  int rcode;
  std::error_code ec;
  std::string output, errors;
  const auto time_ms = measure<>::execution([&]() {
    using namespace boost::process;
    ipstream stdout, stderr;
    if (!chdir.empty()) {
      rcode = system(cmd, std_out > stdout, std_err > stderr, ec, start_dir = chdir);
    } else {
      rcode = system(cmd, std_out > stdout, std_err > stderr, ec);
    }
    std::ostringstream ss_output, ss_errors;
    stdout >> ss_output.rdbuf();
    stderr >> ss_errors.rdbuf();
    output = ss_output.str();
    errors = ss_errors.str();
  });
  if (rcode || ec) {
    if (log_failure) {
      LOG(ERROR) << "failed cmd: " << cmd;
      LOG(ERROR) << "exit code: " << rcode;
      LOG(ERROR) << "error code: " << ec.value() << " - " << ec.message();
      LOG(ERROR) << "stdout: " << output;
      LOG(ERROR) << "stderr: " << errors;
    }
#if defined(__APPLE__)
    // osx bsdtar options "--use-compress-program" and "--fast-read" together
    // run into pipe write error after tar extracts the first occurrence of a
    // file and closes the read end while the decompression program still writes
    // to the pipe. bsdtar doesn't handle this situation well like gnu tar does.
    if (1 == rcode && cmd.find("--fast-read") &&
        (errors.find("cannot write decoded block") != std::string::npos ||
         errors.find("Broken pipe") != std::string::npos)) {
      // ignore this error, or lose speed advantage of "--fast-read" on osx.
      LOG(ERROR) << "tar error ignored on osx for --fast-read";
    } else
#endif
      // circumvent tar warning on reading file that is "changed as we read it".
      // this warning results from reading a table file under concurrent inserts
      if (1 == rcode && errors.find("changed as we read") != std::string::npos) {
        LOG(ERROR) << "tar error ignored under concurrent inserts";
      } else {
        int error_code;
        std::string error_message;
        if (ec) {
          error_code = ec.value();
          error_message = ec.message();
        } else {
          error_code = rcode;
          // Show a more concise message for permission errors instead of the default
          // verbose message. Error logs will still contain all details.
          if (to_lower(errors).find("permission denied") != std::string::npos) {
            error_message = "Insufficient file read/write permission.";
          } else {
            error_message = errors;
          }
        }
        throw std::runtime_error(
            "An error occurred while executing an internal command. Error code: " +
            std::to_string(error_code) + ", message: " + error_message);
      }
  } else {
    VLOG(3) << "finished cmd: " << cmd;
    VLOG(3) << "time: " << time_ms << " ms";
    VLOG(3) << "stdout: " << output;
  }
  return output;
}

inline std::string simple_file_cat(const std::string& archive_path,
                                   const std::string& file_name,
                                   const std::string& compression,
                                   const bool log_failure = true) {
  ddl_utils::validate_allowed_file_path(archive_path,
                                        ddl_utils::DataTransferType::IMPORT);
#if defined(__APPLE__)
  constexpr static auto opt_occurrence = "--fast-read";
#else
  constexpr static auto opt_occurrence = "--occurrence=1";
#endif
  boost::filesystem::path temp_dir =
      boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
  boost::filesystem::create_directories(temp_dir);
  run("tar " + compression + " -xvf " + get_quoted_string(archive_path) + " " +
          opt_occurrence + " " + file_name,
      temp_dir.string(),
      log_failure);
  const auto output = run("cat " + (temp_dir / file_name).string());
  boost::filesystem::remove_all(temp_dir);
  return output;
}

inline std::string get_table_schema(const std::string& archive_path,
                                    const std::string& table,
                                    const std::string& compression) {
  const auto schema_str =
      simple_file_cat(archive_path, table_schema_filename, compression);
  std::regex regex("@T");
  return std::regex_replace(schema_str, regex, table);
}

// If a table was altered there may be a mapping from old column ids to new ones these
// values need to be replaced in the page headers.
void update_or_drop_column_ids_in_page_headers(
    const boost::filesystem::path& path,
    const std::unordered_map<int, int>& column_ids_map,
    const int32_t table_epoch,
    const bool drop_not_update) {
  const std::string file_path = path.string();
  const std::string file_name = path.filename().string();
  std::vector<std::string> tokens;
  boost::split(tokens, file_name, boost::is_any_of("."));

  // ref. FileMgr::init for hint of data file name layout
  if (tokens.size() <= 2 || !(DATA_FILE_EXT == "." + tokens[2] || tokens[2] == "mapd")) {
    // We are only interested in files in the form <id>.<page_size>.<DATA_FILE_EXT>
    return;
  }

  const auto page_size = boost::lexical_cast<int64_t>(tokens[1]);
  const auto file_size = boost::filesystem::file_size(file_path);
  std::unique_ptr<FILE, decltype(simple_file_closer)> fp(
      std::fopen(file_path.c_str(), "r+"), simple_file_closer);
  if (!fp) {
    throw std::runtime_error("Failed to open " + file_path +
                             " for update: " + std::strerror(errno));
  }
  // TODO(Misiu): Rather than reference an exernal layout we should de-duplicate this
  // page-reading code in a single location.  This will also reduce the need for comments
  // below.
  // ref. FileInfo::openExistingFile for hint of chunk header layout
  for (size_t page = 0; page < file_size / page_size; ++page) {
    int32_t header_info[8];
    if (0 != std::fseek(fp.get(), page * page_size, SEEK_SET)) {
      throw std::runtime_error("Failed to seek to page# " + std::to_string(page) +
                               file_path + " for read: " + std::strerror(errno));
    }
    if (1 != fread(header_info, sizeof header_info, 1, fp.get())) {
      throw std::runtime_error("Failed to read " + file_path + ": " +
                               std::strerror(errno));
    }
    if (const auto header_size = header_info[0]; header_size > 0) {
      // header_info[1] is the page's db_id; but can also be used as an "is deleted"
      // indicator if negative.
      auto& contingent = header_info[1];
      // header_info[2] is the page's table_id; but can also used to store the page's
      // epoch since the FileMgr stores table_id information separately.
      auto& epoch = header_info[2];
      auto& col_id = header_info[3];
      if (File_Namespace::is_page_deleted_with_checkpoint(
              table_epoch, epoch, contingent)) {
        continue;
      }
      auto column_map_it = column_ids_map.find(col_id);
      bool rewrite_header = false;
      if (drop_not_update) {
        // if the header contains a column ID that is a key of the map
        // erase the entire header so that column is effectively dropped
        // the value of the map is ignored, thus allowing us to use the
        // same function for both operations
        if (column_map_it != column_ids_map.end()) {
          // clear the entire header
          std::memset(header_info, 0, sizeof(header_info));
          rewrite_header = true;
        }
      } else {
        if (column_map_it == column_ids_map.end()) {
          throw std::runtime_error("Page " + std::to_string(page) + " in " + file_path +
                                   " has unexpected Column ID " + std::to_string(col_id) +
                                   ". Dump may be corrupt.");
        }
        // If a header contains a column id that is remapped to new location
        // then write that change to the file.
        if (const auto dest_col_id = column_map_it->second; col_id != dest_col_id) {
          col_id = dest_col_id;
          rewrite_header = true;
        }
      }
      if (rewrite_header) {
        if (0 != std::fseek(fp.get(), page * page_size, SEEK_SET)) {
          throw std::runtime_error("Failed to seek to page# " + std::to_string(page) +
                                   file_path + " for write: " + std::strerror(errno));
        }
        if (1 != fwrite(header_info, sizeof header_info, 1, fp.get())) {
          throw std::runtime_error("Failed to write " + file_path + ": " +
                                   std::strerror(errno));
        }
      }
    }
  }
}

// Rewrite column ids in chunk keys in a table's data files under a temp_data_dir,
// including files of all shards of the table. Can be slow for big files but should
// be scale faster than refragmentizing. Table altering should be rare for olap.
// Also used to erase page headers for columns that must be dropped completely.
void update_or_drop_column_ids_in_table_files(
    const int32_t table_epoch,
    const std::string& temp_data_dir,
    const std::unordered_map<int, int>& column_ids_map,
    const bool drop_not_update) {
  boost::filesystem::path base_path(temp_data_dir);
  boost::filesystem::recursive_directory_iterator end_it;
  ThreadController_NS::SimpleThreadController<> thread_controller(cpu_threads());
  for (boost::filesystem::recursive_directory_iterator fit(base_path); fit != end_it;
       ++fit) {
    if (!boost::filesystem::is_symlink(fit->path()) &&
        boost::filesystem::is_regular_file(fit->status())) {
      thread_controller.startThread(update_or_drop_column_ids_in_page_headers,
                                    fit->path(),
                                    column_ids_map,
                                    table_epoch,
                                    drop_not_update);
      thread_controller.checkThreadsStatus();
    }
  }
  thread_controller.finish();
}

void delete_old_symlinks(const std::string& table_data_dir) {
  std::vector<boost::filesystem::path> symlinks;
  for (boost::filesystem::directory_iterator it(table_data_dir), end_it; it != end_it;
       it++) {
    if (boost::filesystem::is_symlink(it->path())) {
      symlinks.emplace_back(it->path());
    }
  }
  for (const auto& symlink : symlinks) {
    boost::filesystem::remove_all(symlink);
  }
}

void add_data_file_symlinks(const std::string& table_data_dir) {
  std::map<boost::filesystem::path, boost::filesystem::path> old_to_new_paths;
  for (boost::filesystem::directory_iterator it(table_data_dir), end_it; it != end_it;
       it++) {
    const auto path = boost::filesystem::canonical(it->path());
    if (path.extension().string() == DATA_FILE_EXT) {
      auto old_path = path;
      old_path.replace_extension(File_Namespace::kLegacyDataFileExtension);
      // Add a symlink to data file, if one does not exist.
      if (!boost::filesystem::exists(old_path)) {
        old_to_new_paths[old_path] = path;
      }
    }
  }
  for (const auto& [old_path, new_path] : old_to_new_paths) {
    boost::filesystem::create_symlink(new_path.filename(), old_path);
  }
}

void rename_table_directories(const File_Namespace::GlobalFileMgr* global_file_mgr,
                              const std::string& temp_data_dir,
                              const std::vector<std::string>& target_paths,
                              const std::string& name_prefix) {
  boost::filesystem::path base_path(temp_data_dir);
  boost::filesystem::directory_iterator end_it;
  int target_path_index = 0;
  for (boost::filesystem::directory_iterator fit(base_path); fit != end_it; ++fit) {
    if (!boost::filesystem::is_regular_file(fit->status())) {
      const std::string file_path = fit->path().string();
      const std::string file_name = fit->path().filename().string();
      if (boost::istarts_with(file_name, name_prefix)) {
        const std::string target_path =
            abs_path(global_file_mgr) + "/" + target_paths[target_path_index++];
        if (std::rename(file_path.c_str(), target_path.c_str())) {
          throw std::runtime_error("Failed to rename file " + file_path + " to " +
                                   target_path + ": " + std::strerror(errno));
        }
        // Delete any old/invalid symlinks contained in table dump.
        delete_old_symlinks(target_path);
        File_Namespace::FileMgr::renameAndSymlinkLegacyFiles(target_path);
        // For post-rebrand table dumps, symlinks need to be added here, since file mgr
        // migration would already have been executed for the dumped table.
        add_data_file_symlinks(target_path);
      }
    }
  }
}

std::unordered_map<int, int> find_render_group_columns(
    const std::list<ColumnDescriptor>& src_columns,
    std::vector<std::string>& src_oldinfo_strs,
    const std::string& archive_path) {
  // scan for poly or mpoly columns and collect their names
  std::vector<std::string> poly_column_names;
  for (auto const& src_column : src_columns) {
    auto const sqltype = src_column.columnType.get_type();
    if (sqltype == kPOLYGON || sqltype == kMULTIPOLYGON) {
      poly_column_names.push_back(src_column.columnName);
    }
  }

  // remove any matching render group columns from the source list
  // and capture their IDs in the keys of a map (value is ignored)
  std::unordered_map<int, int> column_ids_to_drop;
  auto last_itr = std::remove_if(
      src_oldinfo_strs.begin(),
      src_oldinfo_strs.end(),
      [&](const std::string& v) -> bool {
        // tokenize
        std::vector<std::string> tokens;
        boost::algorithm::split(
            tokens, v, boost::is_any_of(":"), boost::token_compress_on);
        // extract name and ID
        if (tokens.size() < 2) {
          throw std::runtime_error(
              "Dump " + archive_path +
              " has invalid oldinfo file contents. Dump may be corrupt.");
        }
        auto const& column_name = tokens[0];
        auto const column_id = std::stoi(tokens[1]);
        for (auto const& poly_column_name : poly_column_names) {
          // is it a render group column?
          auto const render_group_column_name = poly_column_name + "_render_group";
          if (column_name == render_group_column_name) {
            LOG(INFO) << "RESTORE TABLE dropping render group column '"
                      << render_group_column_name << "' from dump " << archive_path;
            // add to "set"
            column_ids_to_drop[column_id] = -1;
            return true;
          }
        }
        return false;
      });
  src_oldinfo_strs.erase(last_itr, src_oldinfo_strs.end());

  return column_ids_to_drop;
}

void drop_render_group_columns(
    const std::unordered_map<int, int>& render_group_column_ids,
    const std::string& archive_path,
    const std::string& temp_data_dir,
    const std::string& compression) {
  // rewrite page files to drop the columns with IDs that are the keys of the map
  if (render_group_column_ids.size()) {
    const auto epoch = boost::lexical_cast<int32_t>(
        simple_file_cat(archive_path, table_epoch_filename, compression));
    const auto time_ms = measure<>::execution([&]() {
      update_or_drop_column_ids_in_table_files(
          epoch, temp_data_dir, render_group_column_ids, true /* drop */);
    });
    VLOG(3) << "drop render group columns: " << time_ms << " ms";
  }
}

}  // namespace

void TableArchiver::dumpTable(const TableDescriptor* td,
                              const std::string& archive_path,
                              const std::string& compression) {
  if (td->is_system_table) {
    throw std::runtime_error("Dumping a system table is not supported.");
  }
  if (td->isForeignTable()) {
    throw std::runtime_error("Dumping a foreign table is not supported.");
  }
  ddl_utils::validate_allowed_file_path(archive_path,
                                        ddl_utils::DataTransferType::EXPORT);
  if (g_cluster) {
    throw std::runtime_error("DUMP/RESTORE is not supported yet on distributed setup.");
  }
  if (boost::filesystem::exists(archive_path)) {
    throw std::runtime_error("Archive " + archive_path + " already exists.");
  }
  if (td->isView || td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    throw std::runtime_error("Dumping view or temporary table is not supported.");
  }
  // create a unique uuid for this table dump
  std::string uuid = boost::uuids::to_string(boost::uuids::random_generator()());

  // collect paths of files to archive
  const auto global_file_mgr = cat_->getDataMgr().getGlobalFileMgr();
  std::vector<std::string> file_paths;
  auto file_writer = [&file_paths, uuid](const std::string& file_name,
                                         const std::string& file_type,
                                         const std::string& file_data) {
    std::unique_ptr<FILE, decltype(simple_file_closer)> fp(
        std::fopen(file_name.c_str(), "w"), simple_file_closer);
    if (!fp) {
      throw std::runtime_error("Failed to create " + file_type + " file '" + file_name +
                               "': " + std::strerror(errno));
    }
    if (std::fwrite(file_data.data(), 1, file_data.size(), fp.get()) < file_data.size()) {
      throw std::runtime_error("Failed to write " + file_type + " file '" + file_name +
                               "': " + std::strerror(errno));
    }
    file_paths.push_back(uuid / std::filesystem::path(file_name).filename());
  };

  const auto file_mgr_dir = std::filesystem::path(abs_path(global_file_mgr));
  const auto uuid_dir = file_mgr_dir / uuid;

  if (!std::filesystem::create_directory(uuid_dir)) {
    throw std::runtime_error("Failed to create work directory '" + uuid_dir.string() +
                             "' while dumping table.");
  }

  ScopeGuard cleanup_guard = [&] {
    if (std::filesystem::exists(uuid_dir)) {
      std::filesystem::remove_all(uuid_dir);
    }
  };

  const auto table_name = td->tableName;
  {
    // - gen dumpversion file
    const auto dumpversion_str = std::to_string(kDumpVersion);
    file_writer(
        uuid_dir / table_dumpversion_filename, "table dumpversion", dumpversion_str);
    // - gen schema file
    const auto schema_str = cat_->dumpSchema(td);
    file_writer(uuid_dir / table_schema_filename, "table schema", schema_str);
    // - gen column-old-info file
    const auto cds = cat_->getAllColumnMetadataForTable(td->tableId, true, true, true);
    std::vector<std::string> column_oldinfo;
    std::transform(cds.begin(),
                   cds.end(),
                   std::back_inserter(column_oldinfo),
                   [&](const auto cd) -> std::string {
                     return cd->columnName + ":" + std::to_string(cd->columnId) + ":" +
                            cat_->getColumnDictDirectory(cd);
                   });
    const auto column_oldinfo_str = boost::algorithm::join(column_oldinfo, " ");
    file_writer(uuid_dir / table_oldinfo_filename, "table old info", column_oldinfo_str);
    // - gen table epoch
    const auto epoch = cat_->getTableEpoch(cat_->getCurrentDB().dbId, td->tableId);
    file_writer(uuid_dir / table_epoch_filename, "table epoch", std::to_string(epoch));
    // - collect table data file paths ...
    const auto data_file_dirs = cat_->getTableDataDirectories(td);
    file_paths.insert(file_paths.end(), data_file_dirs.begin(), data_file_dirs.end());
    // - collect table dict file paths ...
    const auto dict_file_dirs = cat_->getTableDictDirectories(td);
    file_paths.insert(file_paths.end(), dict_file_dirs.begin(), dict_file_dirs.end());
    // tar takes time. release cat lock to yield the cat to concurrent CREATE statements.
  }
  // run tar to archive the files ... this may take a while !!
  run("tar " + compression + " --transform=s|" + uuid +
          std::filesystem::path::preferred_separator + "|| -cvf " +
          get_quoted_string(archive_path) + " " + boost::algorithm::join(file_paths, " "),
      file_mgr_dir);
}

// Restore data and dict files of a table from a tgz archive.
void TableArchiver::restoreTable(const Catalog_Namespace::SessionInfo& session,
                                 const TableDescriptor* td,
                                 const std::string& archive_path,
                                 const std::string& compression) {
  ddl_utils::validate_allowed_file_path(archive_path,
                                        ddl_utils::DataTransferType::IMPORT);
  if (g_cluster) {
    throw std::runtime_error("DUMP/RESTORE is not supported yet on distributed setup.");
  }
  if (!boost::filesystem::exists(archive_path)) {
    throw std::runtime_error("Archive " + archive_path + " does not exist.");
  }
  if (td->isView || td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    throw std::runtime_error("Restoring view or temporary table is not supported.");
  }
  // Obtain table schema read lock to prevent modification of the schema during
  // restoration
  const auto table_read_lock =
      lockmgr::TableSchemaLockMgr::getReadLockForTable(*cat_, td->tableName);
  // prevent concurrent inserts into table during restoration
  const auto insert_data_lock =
      lockmgr::InsertDataLockMgr::getWriteLockForTable(*cat_, td->tableName);

  // untar takes time. no grab of cat lock to yield to concurrent CREATE stmts.
  const auto global_file_mgr = cat_->getDataMgr().getGlobalFileMgr();

  // create a unique uuid for this table restore
  std::string uuid = boost::uuids::to_string(boost::uuids::random_generator()());

  const auto uuid_dir = std::filesystem::path(abs_path(global_file_mgr)) / uuid;

  if (!std::filesystem::create_directory(uuid_dir)) {
    throw std::runtime_error("Failed to create work directory '" + uuid_dir.string() +
                             "' while restoring table.");
  }

  ScopeGuard cleanup_guard = [&] {
    if (std::filesystem::exists(uuid_dir)) {
      std::filesystem::remove_all(uuid_dir);
    }
  };

  // dirs where src files are untarred and dst files are backed up
  constexpr static const auto temp_data_basename = "_data";
  constexpr static const auto temp_back_basename = "_back";
  const auto temp_data_dir = uuid_dir / temp_data_basename;
  const auto temp_back_dir = uuid_dir / temp_back_basename;

  // extract & parse schema
  const auto schema_str = get_table_schema(archive_path, td->tableName, compression);
  std::unique_ptr<Parser::Stmt> stmt = Parser::create_stmt_for_query(schema_str, session);
  const auto create_table_stmt = dynamic_cast<Parser::CreateTableStmt*>(stmt.get());
  CHECK(create_table_stmt);

  // verify compatibility between source and destination schemas
  TableDescriptor src_td;
  std::list<ColumnDescriptor> src_columns;
  std::vector<Parser::SharedDictionaryDef> shared_dict_defs;
  create_table_stmt->executeDryRun(session, src_td, src_columns, shared_dict_defs);
  // - sanity check table-level compatibility
  if (src_td.hasDeletedCol != td->hasDeletedCol) {
    // TODO: allow the case, in which src data enables vacuum while
    // dst doesn't, by simply discarding src $deleted column data.
    throw std::runtime_error("Incompatible table VACCUM option");
  }
  if (src_td.nShards != td->nShards) {
    // TODO: allow different shard numbers if they have a "GCD",
    // by splitting/merging src data files before drop into dst.
    throw std::runtime_error("Unmatched number of table shards");
  }
  // - sanity check column-level compatibility (based on column names)
  const auto dst_columns =
      cat_->getAllColumnMetadataForTable(td->tableId, false, false, false);
  if (dst_columns.size() != src_columns.size()) {
    throw std::runtime_error("Unmatched number of table columns");
  }
  for (const auto& [src_cd, dst_cd] : boost::combine(src_columns, dst_columns)) {
    if (src_cd.columnType.get_type_name() != dst_cd->columnType.get_type_name() ||
        src_cd.columnType.get_compression_name() !=
            dst_cd->columnType.get_compression_name()) {
      throw std::runtime_error("Incompatible types on column " + src_cd.columnName);
    }
  }
  // extract src table column ids (ALL columns incl. system/virtual/phy geo cols)
  const auto all_src_oldinfo_str =
      simple_file_cat(archive_path, table_oldinfo_filename, compression);
  std::vector<std::string> src_oldinfo_strs;
  boost::algorithm::split(src_oldinfo_strs,
                          all_src_oldinfo_str,
                          boost::is_any_of(" "),
                          boost::token_compress_on);

  // fetch dump version
  int dump_version = -1;
  try {
    // attempt to read file, do not log if fail to read
    auto const dump_version_str =
        simple_file_cat(archive_path, table_dumpversion_filename, compression, false);
    dump_version = std::stoi(dump_version_str);
  } catch (std::runtime_error& e) {
    // no dump version file found
    dump_version = 0;
  }
  LOG(INFO) << "Dump Version: " << dump_version;

  // version-specific behavior
  const bool do_drop_render_group_columns =
      (dump_version < kDumpVersion_remove_render_group_columns);

  // remove any render group columns from the source columns so that the list of
  // source columns matches the already-created table, and the removed ones will
  // not have an entry in column_ids_map, and hence will not have their data
  // mapped later (effectively dropping them), and return their IDs for when
  // they are actually dropped later
  std::unordered_map<int, int> render_group_column_ids;
  if (do_drop_render_group_columns) {
    render_group_column_ids =
        find_render_group_columns(src_columns, src_oldinfo_strs, archive_path);
  }

  // compare with the destination columns
  auto all_dst_columns =
      cat_->getAllColumnMetadataForTable(td->tableId, true, true, true);
  if (src_oldinfo_strs.size() != all_dst_columns.size()) {
    throw std::runtime_error("Source table has a unmatched number of columns: " +
                             std::to_string(src_oldinfo_strs.size()) + " vs " +
                             std::to_string(all_dst_columns.size()));
  }
  // build a map of src column ids and dst column ids, just in case src table has been
  // ALTERed before and chunk keys of src table needs to be adjusted accordingly.
  // note: this map is used only for the case of migrating a table and not for restoring
  // a table. When restoring a table, the two tables must have the same column ids.
  //
  // also build a map of src dict paths and dst dict paths for relocating src dicts
  std::unordered_map<int, int> column_ids_map;
  std::unordered_map<std::string, std::string> dict_paths_map;
  // sort inputs of transform in lexical order of column names for correct mappings
  std::list<std::vector<std::string>> src_oldinfo_tokens;
  std::transform(
      src_oldinfo_strs.begin(),
      src_oldinfo_strs.end(),
      std::back_inserter(src_oldinfo_tokens),
      [](const auto& src_oldinfo_str) -> auto{
        std::vector<std::string> tokens;
        boost::algorithm::split(
            tokens, src_oldinfo_str, boost::is_any_of(":"), boost::token_compress_on);
        return tokens;
      });
  src_oldinfo_tokens.sort(
      [](const auto& lhs, const auto& rhs) { return lhs[0].compare(rhs[0]) < 0; });
  all_dst_columns.sort(
      [](auto a, auto b) { return a->columnName.compare(b->columnName) < 0; });
  // transform inputs into the maps
  std::transform(src_oldinfo_tokens.begin(),
                 src_oldinfo_tokens.end(),
                 all_dst_columns.begin(),
                 std::inserter(column_ids_map, column_ids_map.end()),
                 [&](const auto& tokens, const auto& cd) -> std::pair<int, int> {
                   VLOG(3) << boost::algorithm::join(tokens, ":") << " ==> "
                           << cd->columnName << ":" << cd->columnId;
                   dict_paths_map[tokens[2]] = cat_->getColumnDictDirectory(cd);
                   return {boost::lexical_cast<int>(tokens[1]), cd->columnId};
                 });
  bool was_table_altered = false;
  std::for_each(column_ids_map.begin(), column_ids_map.end(), [&](auto& it) {
    was_table_altered = was_table_altered || it.first != it.second;
  });
  VLOG(3) << "was_table_altered = " << was_table_altered;

  // extract all data files to a temp dir. will swap with dst table dir after all set,
  // otherwise will corrupt table in case any bad thing happens in the middle.
  run("rm -rf " + temp_data_dir.string());
  run("mkdir -p " + temp_data_dir.string());
  run("tar " + compression + " -xvf " + get_quoted_string(archive_path), temp_data_dir);

  // drop the render group columns here
  if (do_drop_render_group_columns) {
    drop_render_group_columns(
        render_group_column_ids, archive_path, temp_data_dir, compression);
  }

  // if table was ever altered after it was created, update column ids in chunk headers.
  if (was_table_altered) {
    const auto epoch = boost::lexical_cast<int32_t>(
        simple_file_cat(archive_path, table_epoch_filename, compression));
    const auto time_ms = measure<>::execution([&]() {
      update_or_drop_column_ids_in_table_files(
          epoch, temp_data_dir, column_ids_map, false /* update */);
    });
    VLOG(3) << "update_column_ids_table_files: " << time_ms << " ms";
  }

  // finally,,, swap table data/dict dirs!
  const auto data_file_dirs = cat_->getTableDataDirectories(td);
  const auto dict_file_dirs = cat_->getTableDictDirectories(td);
  // move current target dirs, if exists, to backup dir
  std::vector<std::string> both_file_dirs;
  std::merge(data_file_dirs.begin(),
             data_file_dirs.end(),
             dict_file_dirs.begin(),
             dict_file_dirs.end(),
             std::back_inserter(both_file_dirs));
  bool backup_completed = false;
  try {
    run("rm -rf " + temp_back_dir.string());
    run("mkdir -p " + temp_back_dir.string());
    for (const auto& dir : both_file_dirs) {
      const auto dir_full_path = abs_path(global_file_mgr) + "/" + dir;
      if (boost::filesystem::is_directory(dir_full_path)) {
        run("mv " + dir_full_path + " " + temp_back_dir.string());
      }
    }
    backup_completed = true;
    // Move table directories from temp dir to main data directory.
    rename_table_directories(global_file_mgr, temp_data_dir, data_file_dirs, "table_");
    // Move dictionaries from temp dir to main dir.
    for (const auto& dit : dict_paths_map) {
      if (!dit.first.empty() && !dit.second.empty()) {
        const auto src_dict_path = temp_data_dir.string() + "/" + dit.first;
        const auto dst_dict_path = abs_path(global_file_mgr) + "/" + dit.second;
        run("mv " + src_dict_path + " " + dst_dict_path);
      }
    }
    // throw if sanity test forces a rollback
    if (g_test_rollback_dump_restore) {
      throw std::runtime_error("lol!");
    }
  } catch (...) {
    // once backup is completed, whatever in abs_path(global_file_mgr) is the "src"
    // dirs that are to be rolled back and discarded
    if (backup_completed) {
      run("rm -rf " + boost::algorithm::join(both_file_dirs, " "),
          abs_path(global_file_mgr));
    }
    // complete rollback by recovering original "dst" table dirs from backup dir
    boost::filesystem::path base_path(temp_back_dir);
    boost::filesystem::directory_iterator end_it;
    for (boost::filesystem::directory_iterator fit(base_path); fit != end_it; ++fit) {
      run("mv " + fit->path().string() + " .", abs_path(global_file_mgr));
    }
    throw;
  }
  // set for reloading table from the restored/migrated files
  const auto epoch = simple_file_cat(archive_path, table_epoch_filename, compression);
  cat_->setTableEpoch(
      cat_->getCurrentDB().dbId, td->tableId, boost::lexical_cast<int>(epoch));
}

#ifdef HAVE_AWS_S3
namespace {
std::string get_restore_dir_path() {
  auto uuid = boost::uuids::to_string(boost::uuids::random_generator()());
  auto restore_dir_path = std::filesystem::canonical(g_base_path) /
                          shared::kDefaultImportDirName / ("s3-restore-" + uuid);
  return restore_dir_path;
}

std::string download_s3_file(const std::string& s3_archive_path,
                             const TableArchiverS3Options& s3_options,
                             const std::string& restore_dir_path) {
  S3Archive s3_archive{s3_archive_path,
                       s3_options.s3_access_key,
                       s3_options.s3_secret_key,
                       s3_options.s3_session_token,
                       s3_options.s3_region,
                       s3_options.s3_endpoint,
                       false,
                       std::optional<std::string>{},
                       std::optional<std::string>{},
                       std::optional<std::string>{},
                       restore_dir_path};
  s3_archive.init_for_read();
  const auto& object_key = s3_archive.get_objkeys();
  if (object_key.size() > 1) {
    throw std::runtime_error(
        "S3 URI references multiple files. Only one file can be restored at a time.");
  }
  CHECK_EQ(object_key.size(), size_t(1));
  std::exception_ptr eptr;
  return s3_archive.land(object_key[0], eptr, false, false, false);
}
}  // namespace
#endif

// Migrate a table, which doesn't exist in current db, from a tar ball to the db.
// This actually creates the table and restores data/dict files from the tar ball.
void TableArchiver::restoreTable(const Catalog_Namespace::SessionInfo& session,
                                 const std::string& table_name,
                                 const std::string& archive_path,
                                 const std::string& compression,
                                 const TableArchiverS3Options& s3_options) {
  auto local_archive_path = archive_path;
#ifdef HAVE_AWS_S3
  const auto restore_dir_path = get_restore_dir_path();
  ScopeGuard archive_cleanup_guard = [&archive_path, &restore_dir_path] {
    if (shared::is_s3_uri(archive_path) && std::filesystem::exists(restore_dir_path)) {
      std::filesystem::remove_all(restore_dir_path);
    }
  };
  if (shared::is_s3_uri(archive_path)) {
    local_archive_path = download_s3_file(archive_path, s3_options, restore_dir_path);
  }
#endif

  // replace table name and drop foreign dict references
  const auto schema_str = get_table_schema(local_archive_path, table_name, compression);
  std::unique_ptr<Parser::Stmt> stmt = Parser::create_stmt_for_query(schema_str, session);
  const auto create_table_stmt = dynamic_cast<Parser::CreateTableStmt*>(stmt.get());
  CHECK(create_table_stmt);
  create_table_stmt->execute(session, false /*read-only*/);

  try {
    restoreTable(
        session, cat_->getMetadataForTable(table_name), local_archive_path, compression);
  } catch (...) {
    const auto schema_str = "DROP TABLE IF EXISTS " + table_name + ";";
    std::unique_ptr<Parser::Stmt> stmt =
        Parser::create_stmt_for_query(schema_str, session);
    const auto drop_table_stmt = dynamic_cast<Parser::DropTableStmt*>(stmt.get());
    CHECK(drop_table_stmt);
    drop_table_stmt->execute(session, false /*read-only*/);

    throw;
  }
}

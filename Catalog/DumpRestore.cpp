/*
 * Copyright 2019 OmniSci, Inc.
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
#include <algorithm>
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

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/range/combine.hpp>

#include "Catalog.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "LockMgr/LockMgr.h"
#include "LockMgr/TableLockMgr.h"
#include "Parser/ParseDDL.h"
#include "Parser/ParserNode.h"
#include "Parser/parser.h"
#include "RWLocks.h"
#include "Shared/File.h"
#include "Shared/measure.h"
#include "Shared/thread_count.h"
#include "SharedDictionaryValidator.h"
#include "StringDictionary/StringDictionaryClient.h"
#include "SysCatalog.h"

extern bool g_cluster;
bool g_test_rollback_dump_restore{false};

namespace std {
template <typename T, typename U>
struct tuple_size<boost::tuples::cons<T, U>>
    : boost::tuples::length<boost::tuples::cons<T, U>> {};
template <size_t I, typename T, typename U>
struct tuple_element<I, boost::tuples::cons<T, U>>
    : boost::tuples::element<I, boost::tuples::cons<T, U>> {};
}  // namespace std

namespace Catalog_Namespace {

using cat_read_lock = read_lock<Catalog>;

constexpr static char const* table_schema_filename = "_table.sql";
constexpr static char const* table_oldinfo_filename = "_table.oldinfo";
constexpr static char const* table_epoch_filename = "_table.epoch";

inline auto simple_file_closer = [](FILE* f) { std::fclose(f); };

inline std::string abs_path(const File_Namespace::GlobalFileMgr* global_file_mgr) {
  return boost::filesystem::canonical(global_file_mgr->getBasePath()).string();
}

inline std::string run(const std::string& cmd, const std::string& chdir = "") {
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
    LOG(ERROR) << "failed cmd: " << cmd;
    LOG(ERROR) << "exit code: " << rcode;
    LOG(ERROR) << "error code: " << ec.value() << " - " << ec.message();
    LOG(ERROR) << "stdout: " << output;
    LOG(ERROR) << "stderr: " << errors;
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
      throw std::runtime_error("Failed to run command: " + cmd +
                               "\nexit code: " + std::to_string(rcode) + "\nerrors:\n" +
                               (rcode ? errors : ec.message()));
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
                                   const std::string& compression) {
#if defined(__APPLE__)
  constexpr static auto opt_occurrence = " --fast-read ";
#else
  constexpr static auto opt_occurrence = " --occurrence=1 ";
#endif
  boost::filesystem::path temp_dir =
      boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
  boost::filesystem::create_directories(temp_dir);
  run("tar " + compression + " -xvf \"" + archive_path + "\" " + opt_occurrence +
          file_name,
      temp_dir.string());
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

// get a table's data dirs
std::vector<std::string> Catalog::getTableDataDirectories(
    const TableDescriptor* td) const {
  const auto global_file_mgr = getDataMgr().getGlobalFileMgr();
  std::vector<std::string> file_paths;
  for (auto shard : getPhysicalTablesDescriptors(td)) {
    const auto file_mgr = global_file_mgr->getFileMgr(currentDB_.dbId, shard->tableId);
    boost::filesystem::path file_path(file_mgr->getFileMgrBasePath());
    file_paths.push_back(file_path.filename().string());
  }
  return file_paths;
}

// get a column's dict dir basename
std::string Catalog::getColumnDictDirectory(const ColumnDescriptor* cd) const {
  if ((cd->columnType.is_string() || cd->columnType.is_string_array()) &&
      cd->columnType.get_compression() == kENCODING_DICT &&
      cd->columnType.get_comp_param() > 0) {
    const auto dictId = cd->columnType.get_comp_param();
    const DictRef dictRef(currentDB_.dbId, dictId);
    const auto dit = dictDescriptorMapByRef_.find(dictRef);
    CHECK(dit != dictDescriptorMapByRef_.end());
    CHECK(dit->second);
    boost::filesystem::path file_path(dit->second->dictFolderPath);
    return file_path.filename().string();
  }
  return std::string();
}

// get a table's dict dirs
std::vector<std::string> Catalog::getTableDictDirectories(
    const TableDescriptor* td) const {
  std::vector<std::string> file_paths;
  for (auto cd : getAllColumnMetadataForTable(td->tableId, false, false, true)) {
    auto file_base = getColumnDictDirectory(cd);
    if (!file_base.empty() &&
        file_paths.end() == std::find(file_paths.begin(), file_paths.end(), file_base)) {
      file_paths.push_back(file_base);
    }
  }
  return file_paths;
}

// dump a table's schema, data files and dict files to a tgz ball
void Catalog::dumpTable(const TableDescriptor* td,
                        const std::string& archive_path,
                        const std::string& compression) const {
  if (g_cluster) {
    throw std::runtime_error("DUMP/RESTORE is not supported yet on distributed setup.");
  }
  if (boost::filesystem::exists(archive_path)) {
    throw std::runtime_error("Archive " + archive_path + " already exists.");
  }
  if (td->isView || td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    throw std::runtime_error("Dumping view or temporary table is not supported.");
  }
  // collect paths of files to archive
  const auto global_file_mgr = getDataMgr().getGlobalFileMgr();
  std::vector<std::string> file_paths;
  auto file_writer = [&file_paths, global_file_mgr](const std::string& file_name,
                                                    const std::string& file_type,
                                                    const std::string& file_data) {
    const auto file_path = abs_path(global_file_mgr) + "/" + file_name;
    std::unique_ptr<FILE, decltype(simple_file_closer)> fp(
        std::fopen(file_path.c_str(), "w"), simple_file_closer);
    if (!fp) {
      throw std::runtime_error("Failed to create " + file_type + " file '" + file_path +
                               "': " + std::strerror(errno));
    }
    if (std::fwrite(file_data.data(), 1, file_data.size(), fp.get()) < file_data.size()) {
      throw std::runtime_error("Failed to write " + file_type + " file '" + file_path +
                               "': " + std::strerror(errno));
    }
    file_paths.push_back(file_name);
  };
  // grab table read lock for concurrent SELECT (but would block capped COPY or INSERT)
  auto table_read_lock =
      Lock_Namespace::TableLockMgr::getReadLockForTable(*this, td->tableName);

  // grab catalog read lock only. no need table read or checkpoint lock
  // because want to allow concurrent inserts while this dump proceeds.
  const auto table_name = td->tableName;
  {
    cat_read_lock read_lock(this);
    // - gen schema file
    const auto schema_str = dumpSchema(td);
    file_writer(table_schema_filename, "table schema", schema_str);
    // - gen column-old-info file
    const auto cds = getAllColumnMetadataForTable(td->tableId, true, true, true);
    std::vector<std::string> column_oldinfo;
    std::transform(cds.begin(),
                   cds.end(),
                   std::back_inserter(column_oldinfo),
                   [&](const auto cd) -> std::string {
                     return cd->columnName + ":" + std::to_string(cd->columnId) + ":" +
                            getColumnDictDirectory(cd);
                   });
    const auto column_oldinfo_str = boost::algorithm::join(column_oldinfo, " ");
    file_writer(table_oldinfo_filename, "table old info", column_oldinfo_str);
    // - gen table epoch
    const auto epoch = getTableEpoch(currentDB_.dbId, td->tableId);
    file_writer(table_epoch_filename, "table epoch", std::to_string(epoch));
    // - collect table data file paths ...
    const auto data_file_dirs = getTableDataDirectories(td);
    file_paths.insert(file_paths.end(), data_file_dirs.begin(), data_file_dirs.end());
    // - collect table dict file paths ...
    const auto dict_file_dirs = getTableDictDirectories(td);
    file_paths.insert(file_paths.end(), dict_file_dirs.begin(), dict_file_dirs.end());
    // tar takes time. release cat lock to yield the cat to concurrent CREATE statements.
  }
  // run tar to archive the files ... this may take a while !!
  run("tar " + compression + " -cvf \"" + archive_path + "\" " +
          boost::algorithm::join(file_paths, " "),
      abs_path(global_file_mgr));
}

// returns table schema in a string
std::string Catalog::dumpSchema(const TableDescriptor* td) const {
  std::ostringstream os;
  os << "CREATE TABLE @T (";
  // gather column defines
  const auto cds = getAllColumnMetadataForTable(td->tableId, false, false, false);
  std::string comma;
  std::vector<std::string> shared_dicts;
  std::map<const std::string, const ColumnDescriptor*> dict_root_cds;
  for (const auto cd : cds) {
    if (!(cd->isSystemCol || cd->isVirtualCol)) {
      const auto& ti = cd->columnType;
      os << comma << cd->columnName;
      // CHAR is perculiar... better dump it as TEXT(32) like \d does
      if (ti.get_type() == SQLTypes::kCHAR) {
        os << " "
           << "TEXT";
      } else if (ti.get_subtype() == SQLTypes::kCHAR) {
        os << " "
           << "TEXT[]";
      } else {
        os << " " << ti.get_type_name();
      }
      os << (ti.get_notnull() ? " NOT NULL" : "");
      if (ti.is_string()) {
        if (ti.get_compression() == kENCODING_DICT) {
          // if foreign reference, get referenced tab.col
          const auto dict_id = ti.get_comp_param();
          const DictRef dict_ref(currentDB_.dbId, dict_id);
          const auto dict_it = dictDescriptorMapByRef_.find(dict_ref);
          CHECK(dict_it != dictDescriptorMapByRef_.end());
          const auto dict_name = dict_it->second->dictName;
          // when migrating a table, any foreign dict ref will be dropped
          // and the first cd of a dict will become root of the dict
          if (dict_root_cds.end() == dict_root_cds.find(dict_name)) {
            dict_root_cds[dict_name] = cd;
            os << " ENCODING " << ti.get_compression_name() << "(" << (ti.get_size() * 8)
               << ")";
          } else {
            const auto dict_root_cd = dict_root_cds[dict_name];
            shared_dicts.push_back("SHARED DICTIONARY (" + cd->columnName +
                                   ") REFERENCES @T(" + dict_root_cd->columnName + ")");
            // "... shouldn't specify an encoding, it borrows from the referenced column"
          }
        } else {
          os << " ENCODING NONE";
        }
      } else if (ti.get_size() > 0 && ti.get_size() != ti.get_logical_size()) {
        const auto comp_param = ti.get_comp_param() ? ti.get_comp_param() : 32;
        os << " ENCODING " << ti.get_compression_name() << "(" << comp_param << ")";
      }
      comma = ", ";
    }
  }
  // gather SHARED DICTIONARYs
  if (shared_dicts.size()) {
    os << ", " << boost::algorithm::join(shared_dicts, ", ");
  }
  // gather WITH options ...
  std::vector<std::string> with_options;
  with_options.push_back("FRAGMENT_SIZE=" + std::to_string(td->maxFragRows));
  with_options.push_back("MAX_CHUNK_SIZE=" + std::to_string(td->maxChunkSize));
  with_options.push_back("PAGE_SIZE=" + std::to_string(td->fragPageSize));
  with_options.push_back("MAX_ROWS=" + std::to_string(td->maxRows));
  with_options.emplace_back(td->hasDeletedCol ? "VACUUM='DELAYED'"
                                              : "VACUUM='IMMEDIATE'");
  if (!td->partitions.empty()) {
    with_options.push_back("PARTITIONS='" + td->partitions + "'");
  }
  if (td->nShards > 0) {
    const auto shard_cd = getMetadataForColumn(td->tableId, td->shardedColumnId);
    CHECK(shard_cd);
    os << ", SHARD KEY(" << shard_cd->columnName << ")";
    with_options.push_back("SHARD_COUNT=" + std::to_string(td->nShards));
  }
  if (td->sortedColumnId > 0) {
    const auto sort_cd = getMetadataForColumn(td->tableId, td->sortedColumnId);
    CHECK(sort_cd);
    with_options.push_back("SORT_COLUMN='" + sort_cd->columnName + "'");
  }
  os << ") WITH (" + boost::algorithm::join(with_options, ", ") + ");";
  return os.str();
}

// Adjust column ids in chunk keys in a table's data files under a temp_data_dir,
// including files of all shards of the table. Can be slow for big files but should
// be scale faster than refragmentizing. Table altering should be rare for olap.
void Catalog::adjustAlteredTableFiles(
    const std::string& temp_data_dir,
    const std::unordered_map<int, int>& column_ids_map) const {
  boost::filesystem::path base_path(temp_data_dir);
  boost::filesystem::recursive_directory_iterator end_it;
  ThreadController_NS::SimpleThreadController<> thread_controller(cpu_threads());
  for (boost::filesystem::recursive_directory_iterator fit(base_path); fit != end_it;
       ++fit) {
    if (boost::filesystem::is_regular_file(fit->status())) {
      const std::string file_path = fit->path().string();
      const std::string file_name = fit->path().filename().string();
      std::vector<std::string> tokens;
      boost::split(tokens, file_name, boost::is_any_of("."));
      // ref. FileMgr::init for hint of data file name layout
      if (tokens.size() > 2 && MAPD_FILE_EXT == "." + tokens[2]) {
        thread_controller.startThread([file_name, file_path, tokens, &column_ids_map] {
          const auto page_size = boost::lexical_cast<int64_t>(tokens[1]);
          const auto file_size = boost::filesystem::file_size(file_path);
          std::unique_ptr<FILE, decltype(simple_file_closer)> fp(
              std::fopen(file_path.c_str(), "r+"), simple_file_closer);
          if (!fp) {
            throw std::runtime_error("Failed to open " + file_path +
                                     " for update: " + std::strerror(errno));
          }
          // ref. FileInfo::openExistingFile for hint of chunk header layout
          for (size_t page = 0; page < file_size / page_size; ++page) {
            int ints[8];
            if (0 != std::fseek(fp.get(), page * page_size, SEEK_SET)) {
              throw std::runtime_error("Failed to seek to page# " + std::to_string(page) +
                                       file_path + " for read: " + std::strerror(errno));
            }
            if (1 != fread(ints, sizeof ints, 1, fp.get())) {
              throw std::runtime_error("Failed to read " + file_path + ": " +
                                       std::strerror(errno));
            }
            if (ints[0] > 0) {  // header size
              auto cit = column_ids_map.find(ints[3]);
              CHECK(cit != column_ids_map.end());
              if (ints[3] != cit->second) {
                ints[3] = cit->second;
                if (0 != std::fseek(fp.get(), page * page_size, SEEK_SET)) {
                  throw std::runtime_error("Failed to seek to page# " +
                                           std::to_string(page) + file_path +
                                           " for write: " + std::strerror(errno));
                }
                if (1 != fwrite(ints, sizeof ints, 1, fp.get())) {
                  throw std::runtime_error("Failed to write " + file_path + ": " +
                                           std::strerror(errno));
                }
              }
            }
          }
        });
        thread_controller.checkThreadsStatus();
      }
    }
  }
  thread_controller.finish();
}

// Rename table data directories in temp_data_dir to those in target_paths.
// Note: It applies to table migration and not to table recovery.
void Catalog::renameTableDirectories(const std::string& temp_data_dir,
                                     const std::vector<std::string>& target_paths,
                                     const std::string& name_prefix) const {
  const auto global_file_mgr = getDataMgr().getGlobalFileMgr();
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
      }
    }
  }
}

// Restore data and dict files of a table from a tgz ball.
void Catalog::restoreTable(const SessionInfo& session,
                           const TableDescriptor* td,
                           const std::string& archive_path,
                           const std::string& compression) {
  if (g_cluster) {
    throw std::runtime_error("DUMP/RESTORE is not supported yet on distributed setup.");
  }
  if (!boost::filesystem::exists(archive_path)) {
    throw std::runtime_error("Archive " + archive_path + " does not exist.");
  }
  if (td->isView || td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    throw std::runtime_error("Restoring view or temporary table is not supported.");
  }
  // should get checkpoint lock to block any data injection which is meaningless
  // once after the table is restored from data in the source table files.
  auto checkpoint_lock =
      Lock_Namespace::getTableLock<mapd_shared_mutex, mapd_unique_lock>(
          *this, td->tableName, Lock_Namespace::LockType::CheckpointLock);
  // grab table read lock for concurrent SELECT
  auto table_read_lock =
      Lock_Namespace::TableLockMgr::getReadLockForTable(*this, td->tableName);
  // untar takes time. no grab of cat lock to yield to concurrent CREATE stmts.
  const auto global_file_mgr = getDataMgr().getGlobalFileMgr();
  // dirs where src files are untarred and dst files are backed up
  constexpr static const auto temp_data_basename = "_data";
  constexpr static const auto temp_back_basename = "_back";
  const auto temp_data_dir = abs_path(global_file_mgr) + "/" + temp_data_basename;
  const auto temp_back_dir = abs_path(global_file_mgr) + "/" + temp_back_basename;
  // clean up tmp dirs and files in any case
  auto tmp_files_cleaner = [&](void*) {
    run("rm -rf " + temp_data_dir + " " + temp_back_dir);
    run("rm -f " + abs_path(global_file_mgr) + "/" + table_schema_filename);
    run("rm -f " + abs_path(global_file_mgr) + "/" + table_oldinfo_filename);
    run("rm -f " + abs_path(global_file_mgr) + "/" + table_epoch_filename);
  };
  std::unique_ptr<decltype(tmp_files_cleaner), decltype(tmp_files_cleaner)> tfc(
      &tmp_files_cleaner, tmp_files_cleaner);
  // extract & parse schema
  const auto schema_str = get_table_schema(archive_path, td->tableName, compression);
  const auto create_table_stmt =
      Parser::parseDDL<Parser::CreateTableStmt>("table schema", schema_str);
  // verify compatibility between source and destination schemas
  TableDescriptor src_td;
  std::list<ColumnDescriptor> src_columns;
  std::vector<SharedDictionaryDef> shared_dict_defs;
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
  const auto dst_columns = getAllColumnMetadataForTable(td->tableId, false, false, false);
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
  auto all_dst_columns = getAllColumnMetadataForTable(td->tableId, true, true, true);
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
      [](const auto& src_oldinfo_str) -> auto {
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
                   dict_paths_map[tokens[2]] = getColumnDictDirectory(cd);
                   return {boost::lexical_cast<int>(tokens[1]), cd->columnId};
                 });
  bool was_table_altered = false;
  std::for_each(column_ids_map.begin(), column_ids_map.end(), [&](auto& it) {
    was_table_altered = was_table_altered || it.first != it.second;
  });
  VLOG(3) << "was_table_altered = " << was_table_altered;
  // extract all data files to a temp dir. will swap with dst table dir after all set,
  // otherwise will corrupt table in case any bad thing happens in the middle.
  run("rm -rf " + temp_data_dir);
  run("mkdir -p " + temp_data_dir);
  run("tar " + compression + " -xvf \"" + archive_path + "\"", temp_data_dir);
  // if table was ever altered after it was created, update column ids in chunk headers.
  if (was_table_altered) {
    const auto time_ms = measure<>::execution(
        [&]() { adjustAlteredTableFiles(temp_data_dir, column_ids_map); });
    VLOG(3) << "adjustAlteredTableFiles: " << time_ms << " ms";
  }
  // finally,,, swap table data/dict dirs!
  const auto data_file_dirs = getTableDataDirectories(td);
  const auto dict_file_dirs = getTableDictDirectories(td);
  // move current target dirs, if exists, to backup dir
  std::vector<std::string> both_file_dirs;
  std::merge(data_file_dirs.begin(),
             data_file_dirs.end(),
             dict_file_dirs.begin(),
             dict_file_dirs.end(),
             std::back_inserter(both_file_dirs));
  bool backup_completed = false;
  // protect table schema and quickly swap table files
  cat_read_lock read_lock(this);
  try {
    run("rm -rf " + temp_back_dir);
    run("mkdir -p " + temp_back_dir);
    for (const auto& dir : both_file_dirs) {
      const auto dir_full_path = abs_path(global_file_mgr) + "/" + dir;
      if (boost::filesystem::is_directory(dir_full_path)) {
        run("mv " + dir_full_path + " " + temp_back_dir);
      }
    }
    backup_completed = true;
    // accord src data dirs to dst
    renameTableDirectories(temp_data_dir, data_file_dirs, "table_");
    // accord src dict dirs to dst
    for (const auto& dit : dict_paths_map) {
      if (!dit.first.empty() && !dit.second.empty()) {
        const auto src_dict_path = temp_data_dir + "/" + dit.first;
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
  setTableEpoch(currentDB_.dbId, td->tableId, boost::lexical_cast<int>(epoch));
}

// Migrate a table, which doesn't exist in current db, from a tar ball to the db.
// This actually creates the table and restores data/dict files from the tar ball.
void Catalog::restoreTable(const SessionInfo& session,
                           const std::string& table_name,
                           const std::string& archive_path,
                           const std::string& compression) {
  // replace table name and drop foreign dict references
  const auto schema_str = get_table_schema(archive_path, table_name, compression);
  Parser::parseDDL<Parser::CreateTableStmt>("table schema", schema_str)->execute(session);
  try {
    restoreTable(session, getMetadataForTable(table_name), archive_path, compression);
  } catch (...) {
    Parser::parseDDL<Parser::DropTableStmt>("statement",
                                            "DROP TABLE IF EXISTS " + table_name + ";")
        ->execute(session);
    throw;
  }
}

}  // namespace Catalog_Namespace

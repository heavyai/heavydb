#ifndef COMMANDFUNCTORS_H
#define COMMANDFUNCTORS_H

#include "CommandResolutionChain.h"
#include "ThriftOps.h"

#include "gen-cpp/MapD.h"

#include "ClientContext.h"  // Provides us with default class
#include "RegexSupport.h"

#include <rapidjson/document.h>
#include <iostream>
#include "Fragmenter/InsertOrderFragmenter.h"
#include "MapDServer.h"

template <typename CONTEXT_OP_POLICY>
class ContextOperations {
 public:
  using ThriftService = typename CONTEXT_OP_POLICY::ThriftServiceType;
  using ContextType = typename CONTEXT_OP_POLICY::ContextType;

  static void import_geo_table(ContextType context) { thrift_op<kIMPORT_GEO_TABLE>(context); }

  static void get_all_roles(ContextType context) {
    context.role_names.clear();
    context.userPrivateRole = false;
    thrift_op<kGET_ALL_ROLES>(context,
                              [](ContextType& lambda_context) {
                                for (auto role_name : lambda_context.role_names) {
                                  std::cout << role_name << std::endl;
                                }
                              },
                              [](ContextType&) { std::cout << "Cannot connect to MapD Server." << std::endl; });
  }

  static void get_all_roles_for_user(ContextType context) {
    context.role_names.clear();
    thrift_op<kGET_ROLES_FOR_USER>(context,
                                   context.privs_user_name.c_str(),
                                   [](ContextType& lambda_context) {
                                     if (lambda_context.role_names.size() == 0) {
                                       std::cout << "No roles are granted to user "
                                                 << lambda_context.privs_user_name.c_str() << std::endl;
                                     } else {
                                       for (auto role_name : lambda_context.role_names) {
                                         std::cout << role_name << std::endl;
                                       }
                                     }
                                   },
                                   [](ContextType&) { std::cout << "Cannot connect to MapD Server." << std::endl; });
  }
};

struct DefaultContextOpPolicy {
  using ThriftServiceType = ThriftService;
  using ContextType = ClientContext;
};

template <typename CONTEXT_OPS_TYPE>
class CmdBase : public CmdDeterminant {
 public:
  using ContextOps = CONTEXT_OPS_TYPE;
  using ContextType = typename ContextOps::ContextType;

  CmdBase(ContextType& context) : m_stashed_context_ref(context) {}

 protected:
  ContextType& cmdContext() { return m_stashed_context_ref; }

  ContextType& m_stashed_context_ref;
};

// The StandardCommand macro removes boilerplate that needs to be written
// for creating functors that perform commands (usually operations on a client
// context)

#define StandardCommand(CommandName, CommandOperations)                             \
  template <typename CONTEXT_OP_POLICY = DefaultContextOpPolicy,                    \
            template <typename> class CONTEXT_OPS_TYPE = ContextOperations>         \
  class CommandName##Cmd : public CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>> {    \
   public:                                                                          \
    using Super = CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>>;                     \
    using ContextOps = typename Super::ContextOps;                                  \
    using ContextType = typename Super::ContextType;                                \
    CommandName##Cmd(typename Super::ContextType& c) : Super(c) {}                  \
    using Super::cmdContext;                                                        \
    template <typename PARAMETERS>                                                  \
    void operator()(PARAMETERS const& p, std::ostream& output_stream = std::cout) { \
      do {                                                                          \
        CommandOperations;                                                          \
      } while (0);                                                                  \
    }                                                                               \
  }

//
// Standard Command Definitions
//

StandardCommand(CopyGeo, {
  cmdContext().file_name = p[1];   // File name is the first parameter
  cmdContext().table_name = p[2];  // Table name is the second parameter
  ContextOps::import_geo_table(cmdContext());
});

StandardCommand(RoleList, {
  cmdContext().privs_user_name.clear();
  cmdContext().privs_user_name = p[1];  // User name is the first parameter
  ContextOps::get_all_roles_for_user(cmdContext());
});

StandardCommand(Roles, ContextOps::get_all_roles(cmdContext()));

StandardCommand(ListUsers, returned_list_regex<kGET_USERS>(p, cmdContext(), output_stream););
StandardCommand(ListTables, returned_list_regex<kGET_PHYSICAL_TABLES>(p, cmdContext(), output_stream));
StandardCommand(ListViews, returned_list_regex<kGET_VIEWS>(p, cmdContext(), output_stream));

StandardCommand(Help, {
  std::cout << "\\u [regex] List all users, optionally matching regex.\n";
  std::cout << "\\l List all databases.\n";
  std::cout << "\\t [regex] List all tables, optionally matching regex.\n";
  std::cout << "\\v [regex] List all views, optionally matching regex.\n";
  std::cout << "\\d <table> List all columns of a table or a view.\n";
  std::cout << "\\c <database> <user> <password>.\n";
  std::cout << "\\o <table> Return a memory optimized schema based on current data distribution in table.\n";
  std::cout << "\\gpu Execute in GPU mode's.\n";
  std::cout << "\\cpu Execute in CPU mode's.\n";
  std::cout << "\\multiline Set multi-line command line mode.\n";
  std::cout << "\\singleline Set single-line command line mode.\n";
  std::cout << "\\historylen <number> Set history buffer size (default 100).\n";
  std::cout << "\\timing Print timing information.\n";
  std::cout << "\\notiming Do not print timing information.\n";
  std::cout << "\\memory_summary Print memory usage summary.\n";
  std::cout << "\\version Print MapD Server version.\n";
  std::cout << "\\copy <file path> <table> Copy data from file to table.\n";
  std::cout << "\\status Get status of the server and the leaf nodes.\n";
  std::cout << "\\export_dashboard <dashboard name>,<filename> Exports a dashboard to a file\n";
  std::cout << "\\import_dashboard <dashboard name>,<filename> Imports a dashboard from a file\n";
  std::cout << "\\roles Reports all roles.\n";
  std::cout << "\\role_check <roleName> Verifies whether role exists.\n";
  std::cout << "\\role_list <userName> Reports all roles granted to user.\n";
  std::cout << "\\privileges {<roleName>|<userName>} Reports all database objects privileges granted to role or "
               "user.\n";
  std::cout
      << "\\object_privileges <object_name> Reports all privileges granted to an object for all roles and users.\n";
  std::cout << "\\q Quit.\n";
  std::cout.flush();
});

StandardCommand(ListDatabases, {
  thrift_op<kGET_DATABASES>(cmdContext(), [&](ContextType& lambda_context) {
    output_stream << "Database | Owner" << std::endl;
    for (auto p : lambda_context.dbinfos_return)
      output_stream << p.db_name << " | " << p.db_owner << '\n';
    output_stream.flush();
  });
});

StandardCommand(ConnectToDB, {
  if (cmdContext().session != INVALID_SESSION_ID) {
    thrift_op<kDISCONNECT>(cmdContext(), [&](ContextType& lambda_context) {
      output_stream << "Disconnected from database " << cmdContext().db_name << std::endl;
    });
  }

  cmdContext().db_name = p[1];
  cmdContext().user_name = p[2];
  cmdContext().passwd = p[3];

  thrift_op<kCONNECT>(cmdContext(), [&](ContextType& lambda_context) {
    output_stream << "User " << lambda_context.user_name << " connected to database " << lambda_context.db_name
                  << std::endl;
  });
});

StandardCommand(ListColumns, {
  decltype(p[1])& table_name(p[1]);

  auto unserialize_key_metainfo = [](const std::string key_metainfo) -> std::vector<std::string> {
    std::vector<std::string> keys_with_spec;
    rapidjson::Document document;
    document.Parse(key_metainfo.c_str());
    CHECK(!document.HasParseError());
    CHECK(document.IsArray());
    for (auto it = document.Begin(); it != document.End(); ++it) {
      const auto& key_with_spec_json = *it;
      CHECK(key_with_spec_json.IsObject());
      const std::string type = key_with_spec_json["type"].GetString();
      const std::string name = key_with_spec_json["name"].GetString();
      auto key_with_spec = type + " (" + name + ")";
      if (type == "SHARED DICTIONARY") {
        key_with_spec += " REFERENCES ";
        const std::string foreign_table = key_with_spec_json["foreign_table"].GetString();
        const std::string foreign_column = key_with_spec_json["foreign_column"].GetString();
        key_with_spec += foreign_table + "(" + foreign_column + ")";
      } else {
        CHECK(type == "SHARD KEY");
      }
      keys_with_spec.push_back(key_with_spec);
    }
    return keys_with_spec;
  };

  auto on_success_lambda = [&](ContextType& context) {
    const auto table_details = context.table_details;
    if (table_details.view_sql.empty()) {
      std::string temp_holder(" ");
      if (table_details.is_temporary) {
        temp_holder = " TEMPORARY ";
      }
      output_stream << "CREATE" + temp_holder + "TABLE " + table_name + " (\n";
    } else {
      output_stream << "CREATE VIEW " + table_name + " AS " + table_details.view_sql << "\n";
      output_stream << "\n"
                    << "View columns:"
                    << "\n\n";
    }
    std::string comma_or_blank("");
    for (TColumnType p : table_details.row_desc) {
      if (p.is_system) {
        continue;
      }
      std::string encoding;
      if (p.col_type.type == TDatumType::STR) {
        encoding = (p.col_type.encoding == 0 ? " ENCODING NONE"
                                             : " ENCODING " + thrift_to_encoding_name(p.col_type) + "(" +
                                                   std::to_string(p.col_type.comp_param) + ")");

      } else {
        encoding = (p.col_type.encoding == 0 ? ""
                                             : " ENCODING " + thrift_to_encoding_name(p.col_type) + "(" +
                                                   std::to_string(p.col_type.comp_param) + ")");
      }
      output_stream << comma_or_blank << p.col_name << " " << thrift_to_name(p.col_type)
                    << (p.col_type.nullable ? "" : " NOT NULL") << encoding;
      comma_or_blank = ",\n";
    }
    if (table_details.view_sql.empty()) {
      const auto keys_with_spec = unserialize_key_metainfo(table_details.key_metainfo);
      for (const auto& key_with_spec : keys_with_spec) {
        output_stream << ",\n" << key_with_spec;
      }
      // push final ")\n";
      output_stream << ")\n";
      comma_or_blank = "";
      std::string frag = "";
      std::string page = "";
      std::string row = "";
      if (DEFAULT_FRAGMENT_ROWS != table_details.fragment_size) {
        frag = "FRAGMENT_SIZE = " + std::to_string(table_details.fragment_size);
        comma_or_blank = ", ";
      }
      if (table_details.shard_count) {
        frag += comma_or_blank + "SHARD_COUNT = " + std::to_string(table_details.shard_count);
        comma_or_blank = ", ";
      }
      if (DEFAULT_PAGE_SIZE != table_details.page_size) {
        page = comma_or_blank + "PAGE_SIZE = " + std::to_string(table_details.page_size);
        comma_or_blank = ", ";
      }
      if (DEFAULT_MAX_ROWS != table_details.max_rows) {
        row = comma_or_blank + "MAX_ROWS = " + std::to_string(table_details.max_rows);
      }
      std::string with = frag + page + row;
      if (with.length() > 0) {
        output_stream << "WITH (" << with << ")\n";
      }
    } else {
      output_stream << "\n";
    }
  };

  thrift_op<kGET_TABLE_DETAILS>(cmdContext(), table_name.c_str(), on_success_lambda);
});

StandardCommand(GetOptimizedSchema, {
  decltype(p[1])& table_name(p[1]);

  auto get_row_count = [](const TQueryResult& query_result) -> size_t {
    CHECK(!query_result.row_set.row_desc.empty());
    if (query_result.row_set.columns.empty()) {
      return 0;
    }
    CHECK_EQ(query_result.row_set.columns.size(), query_result.row_set.row_desc.size());
    return query_result.row_set.columns.front().nulls.size();
  };

  // runs a simple single integer value query and returns that single int value returned
  auto run_query = [get_row_count](ContextType& context, std::string query) -> int {
    thrift_op<kSQL>(context, query.c_str());
    CHECK(get_row_count(context.query_return));
    // std::cerr << "return value is " <<  context.query_return.row_set.columns[0].data.int_col[0];
    return context.query_return.row_set.columns[0].data.int_col[0];
  };

  auto get_optimal_size =
      [run_query](ClientContext& context, std::string table_name, std::string col_name, int col_type) -> int {
    switch (col_type) {
      case TDatumType::STR: {
        int strings = run_query(context, "select count(distinct " + col_name + ") from " + table_name + ";");
        if (strings < pow(2, 8)) {
          return 8;
        } else {
          if (strings < pow(2, 16)) {
            return 16;
          } else {
            return 32;
          }
        }
      }
      case TDatumType::TIME: {
        return 32;
      }
      case TDatumType::DATE:
      case TDatumType::TIMESTAMP: {
        return run_query(context,
                         "select case when (extract( epoch from mn)  > -2147483648 and extract (epoch from mx) < "
                         "2147483647) then 32 else 0 end from (select min(" +
                             col_name + ") mn, max(" + col_name + ") mx from " + table_name + " );");
      }
      case TDatumType::BIGINT: {
        return run_query(
            context,
            "select  case when (mn > -128 and mx < 127) then 8 else case when (mn > -32768 and mx < 32767) "
            "then 16 else case when (mn  > -2147483648 and mx < 2147483647) then 32 else 0 end end end from "
            "(select min(" +
                col_name + ") mn, max(" + col_name + ") mx from " + table_name + " );");
      }
      case TDatumType::INT: {
        return run_query(
            context,
            "select  case when (mn > -128 and mx < 127) then 8 else case when (mn > -32768 and mx < 32767) "
            "then 16 else 0 end end from "
            "(select min(" +
                col_name + ") mn, max(" + col_name + ") mx from " + table_name + " );");
      }
    }
    return 0;
  };

  auto on_success_lambda = [&](ContextType& context) {
    const auto table_details = context.table_details;
    if (table_details.view_sql.empty()) {
      output_stream << "CREATE TABLE " + table_name + " (\n";
    } else {
      std::cerr << "Can't optimize a view, only the underlying tables\n";
      return;
    }
    std::string comma_or_blank("");
    for (TColumnType p : table_details.row_desc) {
      std::string encoding;
      if (p.col_type.type == TDatumType::STR) {
        encoding = (p.col_type.encoding == 0
                        ? " ENCODING NONE"
                        : " ENCODING " + thrift_to_encoding_name(p.col_type) + "(" +
                              std::to_string(get_optimal_size(context, table_name, p.col_name, p.col_type.type)) + ")");

      } else {
        int opt_size = get_optimal_size(context, table_name, p.col_name, p.col_type.type);
        encoding = (opt_size == 0 ? "" : " ENCODING FIXED(" + std::to_string(opt_size) + ")");
      }
      output_stream << comma_or_blank << p.col_name << " " << thrift_to_name(p.col_type)
                    << (p.col_type.nullable ? "" : " NOT NULL") << encoding;
      comma_or_blank = ",\n";
    }
    // push final "\n";
    if (table_details.view_sql.empty()) {
      output_stream << ")\n";
      comma_or_blank = "";
      std::string frag = "";
      std::string page = "";
      std::string row = "";
      if (DEFAULT_FRAGMENT_ROWS != table_details.fragment_size) {
        frag = " FRAGMENT_SIZE = " + std::to_string(table_details.fragment_size);
        comma_or_blank = ",";
      }
      if (DEFAULT_PAGE_SIZE != table_details.page_size) {
        page = comma_or_blank + " PAGE_SIZE = " + std::to_string(table_details.page_size);
        comma_or_blank = ",";
      }
      if (DEFAULT_MAX_ROWS != table_details.max_rows) {
        row = comma_or_blank + " MAX_ROWS = " + std::to_string(table_details.max_rows);
      }
      std::string with = frag + page + row;
      if (with.length() > 0) {
        output_stream << "WITH (" << with << ")\n";
      }
    } else {
      output_stream << "\n";
    }
  };

  thrift_op<kGET_TABLE_DETAILS>(cmdContext(), table_name.c_str(), on_success_lambda);
});

#endif

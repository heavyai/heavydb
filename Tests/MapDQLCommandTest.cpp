#include "../SQLFrontend/CommandFunctors.h"
#include "../SQLFrontend/CommandResolutionChain.h"
#include "../SQLFrontend/MetaClientContext.h"
#include "../SQLFrontend/ThriftOps.h"
#include "../gen-cpp/mapd_types.h"
#include "gtest/gtest.h"

#define MockMethod(method_name) \
  template <typename... ARGS>   \
  void method_name(ARGS&&... args) {}
#define RetValMockMethod(method_name, ret_val) \
  template <typename... ARGS>                  \
  ret_val method_name(ARGS&&... args) {        \
    return ret_val();                          \
  }

struct CoreMockTransport {
  MockMethod(open)
};

// clang-format off
class CoreMockClient {
 public:
  MockMethod(connect)
  MockMethod(disconnect)
  MockMethod(interrupt)
  MockMethod(sql_execute)
  MockMethod(get_tables)
  MockMethod(get_physical_tables)
  MockMethod(get_views)
  MockMethod(get_databases)
  MockMethod(get_users)
  MockMethod(set_execution_mode)
  MockMethod(get_version)
  MockMethod(get_memory)
  MockMethod(get_table_details)
  MockMethod(clear_gpu_memory)
  MockMethod(clear_cpu_memory)
  MockMethod(get_hardware_info)
  MockMethod(import_geo_table)
  MockMethod(set_table_epoch)
  MockMethod(set_table_epoch_by_name)
  RetValMockMethod(get_table_epoch, int32_t)
  RetValMockMethod(get_table_epoch_by_name, int32_t)
  MockMethod(get_status)
  MockMethod(create_frontend_view)
  MockMethod(get_frontend_view)
  MockMethod(get_role)
  MockMethod(get_all_roles)
  MockMethod(get_db_objects_for_role)
  MockMethod(get_db_object_privs)
  MockMethod(get_all_roles_for_user)
  MockMethod(set_license_key)
  MockMethod(get_license_claims)
};
// clang-format on

TEST(MapDQLTest, CommandResolutionChain_Construction_MatchedCommand) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool lambda_called = false;
  CommandResolutionChain<> constructible(
      "\\fake_command token1 token2", "\\fake_command", 3, [&](Params const& p) { lambda_called = true; });

  EXPECT_TRUE(lambda_called);
  EXPECT_TRUE(constructible.is_resolved());
}

TEST(MapDQLTest, CommandResolutionChain_Construction_MismatchedCommand) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool lambda_not_called = true;
  CommandResolutionChain<> constructible(
      "\\fake_command token1 token2", "\\blahblah", 3, [&](Params const& p) { lambda_not_called = false; });

  EXPECT_TRUE(lambda_not_called);
  EXPECT_FALSE(constructible.is_resolved());
}

TEST(MapDQLTest, CommandResolutionChain_DefaultTokenizer) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool lambda_called = false;
  CommandResolutionChain<> constructible(
      "\\fake_command token1 token2", "\\fake_command", 3, [&](Params const& p) { lambda_called = true; });

  EXPECT_EQ(Params::size_type(3), constructible.m_command_token_list.size());
  EXPECT_STREQ("\\fake_command", constructible.m_command_token_list[0].c_str());
  EXPECT_STREQ("token1", constructible.m_command_token_list[1].c_str());
  EXPECT_STREQ("token2", constructible.m_command_token_list[2].c_str());
}

TEST(MapDQLTest, CommandResolutionChain_ThreeSet_FirstHit) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool first_hit = false;
  bool second_hit = false;
  bool third_hit = false;

  auto resolution =
      CommandResolutionChain<>(
          "\\fake_command1 token1 token2", "\\fake_command1", 3, [&](Params const& p) { first_hit = true; })(
          "\\fake_command2", 1, [&](Params const& p) { second_hit = true; })(
          "\\fake_command3", 1, [&](Params const& p) { third_hit = true; })
          .is_resolved();

  EXPECT_TRUE(resolution);
  EXPECT_TRUE(first_hit);
  EXPECT_FALSE(second_hit);
  EXPECT_FALSE(third_hit);
}

TEST(MapDQLTest, CommandResolutionChain_ThreeSet_SecondHit) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool first_hit = false;
  bool second_hit = false;
  bool third_hit = false;

  auto resolution =
      CommandResolutionChain<>(
          "\\fake_command2 token1 token2", "\\fake_command1", 3, [&](Params const& p) { first_hit = true; })(
          "\\fake_command2", 1, [&](Params const& p) { second_hit = true; })(
          "\\fake_command3", 1, [&](Params const& p) { third_hit = true; })
          .is_resolved();

  EXPECT_TRUE(resolution);
  EXPECT_FALSE(first_hit);
  EXPECT_TRUE(second_hit);
  EXPECT_FALSE(third_hit);
}

TEST(MapDQLTest, CommandResolutionChain_ThreeSet_ThirdHit) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool first_hit = false;
  bool second_hit = false;
  bool third_hit = false;

  auto resolution =
      CommandResolutionChain<>(
          "\\fake_command3 token1 token2", "\\fake_command1", 3, [&](Params const& p) { first_hit = true; })(
          "\\fake_command2", 1, [&](Params const& p) { second_hit = true; })(
          "\\fake_command3", 1, [&](Params const& p) { third_hit = true; })
          .is_resolved();

  EXPECT_TRUE(resolution);
  EXPECT_FALSE(first_hit);
  EXPECT_FALSE(second_hit);
  EXPECT_TRUE(third_hit);
}

TEST(MapDQLTest, CommandResolutionChain_ThreeSet_NoHits) {
  using Params = CommandResolutionChain<>::CommandTokenList;

  bool first_hit = false;
  bool second_hit = false;
  bool third_hit = false;

  auto resolution =
      CommandResolutionChain<>(
          "\\i_cant_be_matched token1 token2", "\\fake_command1", 3, [&](Params const& p) { first_hit = true; })(
          "\\fake_command2", 1, [&](Params const& p) { second_hit = true; })(
          "\\fake_command3", 1, [&](Params const& p) { third_hit = true; })
          .is_resolved();

  EXPECT_FALSE(resolution);
  EXPECT_FALSE(first_hit);
  EXPECT_FALSE(second_hit);
  EXPECT_FALSE(third_hit);
}

//
// \\copygeo Command Unit Test and Support Mockups
//

struct CopyGeoCommandMockupContext {
  std::string file_name;
  std::string table_name;
  bool import_geo_table_invoked = false;
};

struct CopyGeoCommandContextOpsPolicy {
  using ThriftServiceType = ThriftService;
  using ContextType = CopyGeoCommandMockupContext;
};

template <typename CONTEXT_OP_POLICY>
class CopyGeoCommandMockupContextOperations {
 public:
  using ThriftService = typename CONTEXT_OP_POLICY::ThriftServiceType;
  using ContextType = typename CONTEXT_OP_POLICY::ContextType;

  static void import_geo_table(ContextType& context) { context.import_geo_table_invoked = true; }
};

TEST(MapDQLTest, CopyGeoCommandTest) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestCopyGeoCmd = CopyGeoCmd<CopyGeoCommandContextOpsPolicy, CopyGeoCommandMockupContextOperations>;
  CopyGeoCommandMockupContext unit_test_context;

  auto resolution = CommandResolutionChain<>(
                        "\\copygeo filename.csv tablename", "\\copygeo", 3, UnitTestCopyGeoCmd(unit_test_context))
                        .is_resolved();

  EXPECT_TRUE(unit_test_context.import_geo_table_invoked);
  EXPECT_TRUE(resolution);
}

//
// \\roles_list Command Unit Test and Support Mockups
//

struct RoleListCommandMockupContext {
  std::string privs_user_name;
  bool get_all_roles_for_user_invoked = false;
};

struct RoleListCommandContextOpsPolicy {
  using ThriftServiceType = ThriftService;
  using ContextType = RoleListCommandMockupContext;
};

template <typename CONTEXT_OP_POLICY>
class RoleListCommandMockupContextOperations {
 public:
  using ThriftService = typename CONTEXT_OP_POLICY::ThriftServiceType;
  using ContextType = typename CONTEXT_OP_POLICY::ContextType;

  static void get_all_roles_for_user(ContextType& context) { context.get_all_roles_for_user_invoked = true; }
};

TEST(MapDQLTest, RoleListCommandTest) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestRoleListCmd = RoleListCmd<RoleListCommandContextOpsPolicy, RoleListCommandMockupContextOperations>;
  RoleListCommandMockupContext unit_test_context;

  auto resolution =
      CommandResolutionChain<>("\\role_list mapd", "\\role_list", 2, UnitTestRoleListCmd(unit_test_context))
          .is_resolved();

  EXPECT_STREQ("mapd", unit_test_context.privs_user_name.c_str());
  EXPECT_TRUE(unit_test_context.get_all_roles_for_user_invoked);
  EXPECT_TRUE(resolution);
}

//
// \\roles Command Unit Test and Support Mockups
//

struct RolesCommandMockupContext {
  bool get_all_roles_invoked = false;
};

struct RolesCommandContextOpsPolicy {
  using ThriftServiceType = ThriftService;
  using ContextType = RolesCommandMockupContext;
};

template <typename CONTEXT_OP_POLICY>
class RolesCommandMockupContextOperations {
 public:
  using ThriftService = typename CONTEXT_OP_POLICY::ThriftServiceType;
  using ContextType = typename CONTEXT_OP_POLICY::ContextType;

  static void get_all_roles(ContextType& context) { context.get_all_roles_invoked = true; }
};

TEST(MapDQLTest, RolesCommandTest) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestRolesCmd = RolesCmd<RolesCommandContextOpsPolicy, RolesCommandMockupContextOperations>;
  RolesCommandMockupContext unit_test_context;

  auto resolution =
      CommandResolutionChain<>("\\roles", "\\roles", 1, UnitTestRolesCmd(unit_test_context)).is_resolved();

  EXPECT_TRUE(unit_test_context.get_all_roles_invoked);
  EXPECT_TRUE(resolution);
}

//
// Mockup Structures Necessary for Testing the Regex Commands \u, \t, \v
//

struct RegexCommandsMockClient : public CoreMockClient {
  template <typename NAMES_RETURN_TYPE, typename SESSION_TYPE>
  void get_users(NAMES_RETURN_TYPE& names_return, SESSION_TYPE const& session) {
    // Fake this response from the server
    names_return = {"mapd", "homer", "marge", "bart", "lisa", "lisaSimpson", "sideshowbob"};
  }

  template <typename NAMES_RETURN_TYPE, typename SESSION_TYPE>
  void get_physical_tables(NAMES_RETURN_TYPE& names_return, SESSION_TYPE const& session) {
    names_return = {"flights_2008_7M", "test", "test_x", "test_inner_x", "bar", "test_inner"};
  }

  template <typename NAMES_RETURN_TYPE, typename SESSION_TYPE>
  void get_views(NAMES_RETURN_TYPE& names_return, SESSION_TYPE const& session) {
    names_return = {"test_view", "cletus", "marge", "marge978", "marge1025"};
  }
};

using RegexCommandsMockupContext = MetaClientContext<RegexCommandsMockClient, CoreMockTransport>;

struct RegexCommandsContextOpsPolicy {
  using ThriftServiceType = ThriftService;
  using ContextType = RegexCommandsMockupContext;
};

template <typename CONTEXT_OP_POLICY>
class RegexCommandsMockupContextOperations {
 public:
  using ThriftService = typename CONTEXT_OP_POLICY::ThriftServiceType;
  using ContextType = typename CONTEXT_OP_POLICY::ContextType;
};

// \u command Unit Tests
//
// See Issue #143, Public Repo for manual test cases replicated in this unit test
//
// Test cases:
//
// \u
// \u ^lisa
// \u ^lisa.+
// \u ^lisa.*

using ListUsersMockClient = RegexCommandsMockClient;
using ListUsersCommandMockupContext = RegexCommandsMockupContext;
using ListUsersCommandContextOpsPolicy = RegexCommandsContextOpsPolicy;

template <typename CONTEXT_OPS_POLICY_TYPE>
using ListUsersCommandMockupContextOperations = RegexCommandsMockupContextOperations<CONTEXT_OPS_POLICY_TYPE>;

TEST(MapDQLTest, ListUsersCommandTest_ListAll) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListUsersCmd = ListUsersCmd<ListUsersCommandContextOpsPolicy, ListUsersCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  // Run the test for \u
  std::ostringstream test_capture_stream1;
  ListUsersCommandMockupContext unit_test_context_list_all;
  auto resolution1 =
      CommandResolutionChain<>("\\u", "\\u", 1, UnitTestListUsersCmd(unit_test_context_list_all), test_capture_stream1)
          .is_resolved();
  EXPECT_TRUE(resolution1);

  std::string output_back_to_input(test_capture_stream1.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  EXPECT_EQ(extractedTokens.size(), (decltype(extractedTokens)::size_type)7);
  EXPECT_EQ(extractedTokens[0], "mapd");
  EXPECT_EQ(extractedTokens[1], "homer");
  EXPECT_EQ(extractedTokens[2], "marge");
  EXPECT_EQ(extractedTokens[3], "bart");
  EXPECT_EQ(extractedTokens[4], "lisa");
  EXPECT_EQ(extractedTokens[5], "lisaSimpson");
  EXPECT_EQ(extractedTokens[6], "sideshowbob");
}

TEST(MapDQLTest, ListUsersCommandTest_OnlyLisa) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListUsersCmd = ListUsersCmd<ListUsersCommandContextOpsPolicy, ListUsersCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream2;
  ListUsersCommandMockupContext unit_test_context_only_lisa;
  auto resolution2 = CommandResolutionChain<>(
                         "\\u ^lisa", "\\u", 1, UnitTestListUsersCmd(unit_test_context_only_lisa), test_capture_stream2)
                         .is_resolved();
  EXPECT_TRUE(resolution2);

  std::string output_back_to_input(test_capture_stream2.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  EXPECT_EQ(extractedTokens.size(), (decltype(extractedTokens)::size_type)1);
  EXPECT_EQ(extractedTokens[0], "lisa");
}

TEST(MapDQLTest, ListUsersCommandTest_StartsWithLisa) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListUsersCmd = ListUsersCmd<ListUsersCommandContextOpsPolicy, ListUsersCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream3;
  ListUsersCommandMockupContext unit_test_context_starts_with_lisa;
  auto resolution3 =
      CommandResolutionChain<>(
          "\\u ^lisa.*", "\\u", 1, UnitTestListUsersCmd(unit_test_context_starts_with_lisa), test_capture_stream3)
          .is_resolved();
  EXPECT_TRUE(resolution3);

  std::string output_back_to_input(test_capture_stream3.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  EXPECT_EQ(extractedTokens.size(), (decltype(extractedTokens)::size_type)2);
  EXPECT_EQ(extractedTokens[0], "lisa");
  EXPECT_EQ(extractedTokens[1], "lisaSimpson");
}

// \t command Unit Tests
//
// See Issue #143, Public Repo for manual test cases replicated in this unit test
//
// Test cases:
//
// \t
// \t _x$
// \t .+_x$
// \t inner
// \t .+inner.+

using ListTablesMockClient = RegexCommandsMockClient;
using ListTablesCommandMockupContext = RegexCommandsMockupContext;
using ListTablesCommandContextOpsPolicy = RegexCommandsContextOpsPolicy;

template <typename CONTEXT_OPS_POLICY_TYPE>
using ListTablesCommandMockupContextOperations = RegexCommandsMockupContextOperations<CONTEXT_OPS_POLICY_TYPE>;

TEST(MapDQLTest, ListTablesCommandTest_ListAll) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListTablesCmd =
      ListTablesCmd<ListTablesCommandContextOpsPolicy, ListTablesCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListTablesCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\t", "\\t", 1, UnitTestListTablesCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;
  EXPECT_EQ(extractedTokens.size(), (TokenCount)6);
  EXPECT_EQ(extractedTokens[0], "flights_2008_7M");
  EXPECT_EQ(extractedTokens[1], "test");
  EXPECT_EQ(extractedTokens[2], "test_x");
  EXPECT_EQ(extractedTokens[3], "test_inner_x");
  EXPECT_EQ(extractedTokens[4], "bar");
  EXPECT_EQ(extractedTokens[5], "test_inner");
}

TEST(MapDQLTest, ListTablesCommandTest_EndsWithNoMatch) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListTablesCmd =
      ListTablesCmd<ListTablesCommandContextOpsPolicy, ListTablesCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListTablesCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\t _x$", "\\t", 1, UnitTestListTablesCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;

  EXPECT_TRUE(extractedTokens.empty());
}

TEST(MapDQLTest, ListTablesCommandTest_EndsWithMatch) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListTablesCmd =
      ListTablesCmd<ListTablesCommandContextOpsPolicy, ListTablesCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListTablesCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\t .+_x$", "\\t", 1, UnitTestListTablesCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;
  EXPECT_EQ(extractedTokens.size(), (TokenCount)2);
  EXPECT_EQ(extractedTokens[0], "test_x");
  EXPECT_EQ(extractedTokens[1], "test_inner_x");
}

TEST(MapDQLTest, ListTablesCommandTest_InnerNoMatch) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListTablesCmd =
      ListTablesCmd<ListTablesCommandContextOpsPolicy, ListTablesCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListTablesCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\t inner", "\\t", 1, UnitTestListTablesCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;
  EXPECT_EQ(extractedTokens.size(), (TokenCount)0);
}

TEST(MapDQLTest, ListTablesCommandTest_InnerMatch) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListTablesCmd =
      ListTablesCmd<ListTablesCommandContextOpsPolicy, ListTablesCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListTablesCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\t .+inner.+", "\\t", 1, UnitTestListTablesCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;
  EXPECT_EQ(extractedTokens.size(), (TokenCount)1);
  EXPECT_EQ(extractedTokens[0], "test_inner_x");
}

// \v command Unit Tests
//
// See Issue #143, Public Repo for manual test cases replicated in this unit test
//
// Test cases:
//
// \v
// \v .+\d{4}$

using ListViewsMockClient = RegexCommandsMockClient;
using ListViewsCommandMockupContext = RegexCommandsMockupContext;
using ListViewsCommandContextOpsPolicy = RegexCommandsContextOpsPolicy;

template <typename CONTEXT_OPS_POLICY_TYPE>
using ListViewsCommandMockupContextOperations = RegexCommandsMockupContextOperations<CONTEXT_OPS_POLICY_TYPE>;

TEST(MapDQLTest, ViewListCommandTest_ListAll) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListViewsCmd = ListViewsCmd<ListViewsCommandContextOpsPolicy, ListViewsCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListViewsCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\v", "\\v", 1, UnitTestListViewsCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;
  EXPECT_EQ(extractedTokens.size(), (TokenCount)5);
  EXPECT_EQ(extractedTokens[0], "test_view");
  EXPECT_EQ(extractedTokens[1], "cletus");
  EXPECT_EQ(extractedTokens[2], "marge");
  EXPECT_EQ(extractedTokens[3], "marge978");
  EXPECT_EQ(extractedTokens[4], "marge1025");
}

TEST(MapDQLTest, ViewListCommandTest_EndsWith4Digits) {
  using Params = CommandResolutionChain<>::CommandTokenList;
  using UnitTestListViewsCmd = ListViewsCmd<ListViewsCommandContextOpsPolicy, ListViewsCommandMockupContextOperations>;
  using TokenExtractor = CommandResolutionChain<>::TokenExtractor;

  std::ostringstream test_capture_stream;
  ListViewsCommandMockupContext unit_test_context;
  auto resolution =
      CommandResolutionChain<>("\\v .+\\d{4}$", "\\v", 1, UnitTestListViewsCmd(unit_test_context), test_capture_stream)
          .is_resolved();
  EXPECT_TRUE(resolution);

  std::string output_back_to_input(test_capture_stream.str());
  auto extractedTokens = TokenExtractor().extract_tokens(output_back_to_input);
  using TokenCount = decltype(extractedTokens)::size_type;
  EXPECT_EQ(extractedTokens.size(), (TokenCount)1);
  EXPECT_EQ(extractedTokens[0], "marge1025");
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

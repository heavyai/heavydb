#include "gtest/gtest.h"
#include "../SQLFrontend/ThriftOps.h"
#include "../SQLFrontend/CommandResolutionChain.h"
#include "../SQLFrontend/CommandFunctors.h"

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
      CommandResolutionChain<>("\\fake_command1 token1 token2", "\\fake_command1", 3, [&](Params const& p) {
        first_hit = true;
      })("\\fake_command2", 1, [&](Params const& p) { second_hit = true; })("\\fake_command3", 1, [&](Params const& p) {
        third_hit = true;
      }).is_resolved();

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
      CommandResolutionChain<>("\\fake_command2 token1 token2", "\\fake_command1", 3, [&](Params const& p) {
        first_hit = true;
      })("\\fake_command2", 1, [&](Params const& p) { second_hit = true; })("\\fake_command3", 1, [&](Params const& p) {
        third_hit = true;
      }).is_resolved();

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
      CommandResolutionChain<>("\\fake_command3 token1 token2", "\\fake_command1", 3, [&](Params const& p) {
        first_hit = true;
      })("\\fake_command2", 1, [&](Params const& p) { second_hit = true; })("\\fake_command3", 1, [&](Params const& p) {
        third_hit = true;
      }).is_resolved();

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
      CommandResolutionChain<>("\\i_cant_be_matched token1 token2", "\\fake_command1", 3, [&](Params const& p) {
        first_hit = true;
      })("\\fake_command2", 1, [&](Params const& p) { second_hit = true; })("\\fake_command3", 1, [&](Params const& p) {
        third_hit = true;
      }).is_resolved();

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

TEST(MapDQLTest, TableListCommandTest) {
  // FIX-ME:  Add after process_backslash_commands is refactored to use the CommandResolutionChain<>
}

TEST(MapDQLTest, UserListCommandTest) {
  // FIX-ME:  Add after process_backslash_commands is refactored to use the CommandResolutionChain<>
}

TEST(MapDQLTest, ViewListCommandTest) {
  // FIX-ME:  Add after process_backslash_commands is refactored to use the CommandResolutionChain<>
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

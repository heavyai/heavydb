#ifndef COMMANDFUNCTORS_H
#define COMMANDFUNCTORS_H

#include "ThriftOps.h"
#include "CommandResolutionChain.h"
#include "gen-cpp/MapD.h"
#include "ClientContext.h"  // Provides us with default class
#include "RegexSupport.h"

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

#endif

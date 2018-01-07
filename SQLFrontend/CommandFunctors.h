#ifndef COMMANDFUNCTORS_H
#define COMMANDFUNCTORS_H

#include "ThriftOps.h"
#include "CommandResolutionChain.h"
#include "gen-cpp/MapD.h"
#include "ClientContext.h"  // Provides us with default class

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
class CmdBase {
 public:
  using ContextOps = CONTEXT_OPS_TYPE;
  using ContextType = typename ContextOps::ContextType;

  CmdBase(ContextType& context) : m_stashed_context_ref(context) {}

 protected:
  ContextType& m_stashed_context_ref;
};

template <typename CONTEXT_OP_POLICY = DefaultContextOpPolicy,
          template <typename> class CONTEXT_OPS_TYPE = ContextOperations>
class CopyGeoCmd : public CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>> {
 public:
  using Super = CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>>;
  CopyGeoCmd(typename Super::ContextType& c) : Super(c) {}

  template <typename PARAMETERS>
  void operator()(PARAMETERS const& p) {
    Super::m_stashed_context_ref.file_name = p[1];
    Super::m_stashed_context_ref.table_name = p[2];

    Super::ContextOps::import_geo_table(Super::m_stashed_context_ref);
  }
};

template <typename CONTEXT_OP_POLICY = DefaultContextOpPolicy,
          template <typename> class CONTEXT_OPS_TYPE = ContextOperations>
class RoleListCmd : public CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>> {
 public:
  using Super = CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>>;
  RoleListCmd(typename Super::ContextType& c) : Super(c) {}

  template <typename PARAMETERS>
  void operator()(PARAMETERS const& p) {
    Super::m_stashed_context_ref.privs_user_name.clear();
    Super::m_stashed_context_ref.privs_user_name = p[1];
    Super::ContextOps::get_all_roles_for_user(Super::m_stashed_context_ref);
  }
};

template <typename CONTEXT_OP_POLICY = DefaultContextOpPolicy,
          template <typename> class CONTEXT_OPS_TYPE = ContextOperations>
class RolesCmd : public CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>> {
 public:
  using Super = CmdBase<CONTEXT_OPS_TYPE<CONTEXT_OP_POLICY>>;
  RolesCmd(typename Super::ContextType& c) : Super(c) {}

  template <typename PARAMETERS>
  void operator()(PARAMETERS const& p) {
    Super::ContextOps::get_all_roles(Super::m_stashed_context_ref);
  }
};

#endif

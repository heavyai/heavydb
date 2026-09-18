/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SamlServer.h"
#include <saml/SAMLConfig.h>
#include <saml/binding/SecurityPolicy.h>
#include <saml/binding/SecurityPolicyRule.h>
#include <saml/saml2/core/Protocols.h>
#include <saml/saml2/metadata/Metadata.h>
#include <saml/saml2/metadata/MetadataCredentialCriteria.h>
#include <saml/saml2/metadata/MetadataProvider.h>
#include <saml/signature/SignatureProfileValidator.h>
#include <xmltooling/XMLObject.h>
#include <xmltooling/XMLObjectBuilder.h>
#include <xmltooling/XMLToolingConfig.h>
#include <xmltooling/security/Credential.h>
#include <xmltooling/security/SignatureTrustEngine.h>
#include <xmltooling/security/TrustEngine.h>
#include <xmltooling/signature/SignatureValidator.h>
#include <xmltooling/util/ParserPool.h>
#include <xmltooling/validation/Validator.h>
#include <boost/algorithm/string/trim.hpp>
#include <fstream>
#include <sstream>
#include "Catalog/Catalog.h"
#include "Logger/Logger.h"
#include "Shared/base64.h"

using namespace opensaml;
using namespace opensaml::saml2;
using namespace opensaml::saml2md;
using namespace xercesc;
using namespace xmltooling;

namespace Catalog_Namespace {
extern bool g_log_user_id;
}

static DOMDocument* get_idp_metadata_provider_xml(const std::string& metadata_path) {
  std::string xml_provider_str =
      R"(<FilesystemMetadataProvider path=")" + metadata_path + R"(" validate="0"/>)";
  std::stringstream ss(xml_provider_str);
  return XMLToolingConfig::getConfig().getParser().parse(ss);
}

SamlServer::SamlServer() : authMetadata_(nullptr) {
  LOG(INFO) << "No SAML SSO server defined, will not attempt to authenticate via SAML";
  samlInUse_ = false;
}

SamlServer::~SamlServer() {
  std::for_each(rules_.begin(), rules_.end(), xmltooling::cleanup<SecurityPolicyRule>());
  if (sp_url_) {
    XMLString::release(&sp_url_);
  }
}

SamlServer::SamlServer(const AuthMetadata& authMetadata)
    : authMetadata_(&authMetadata), samlInUse_(false) {
  samlInUse_ = false;
  if (!authMetadata_->samlIdpMetadataFile.empty() &&
      !authMetadata_->samlSpTargetUrl.empty()) {
    // init opensaml globals
    XMLToolingConfig::getConfig().log_config();
    XMLToolingConfig::getConfig().init();
    SAMLConfig::getConfig().init();
    // init security policy to check ttl and destination
    rules_.push_back(SAMLConfig::getConfig().SecurityPolicyRuleManager.newPlugin(
        CONDITIONS_POLICY_RULE, nullptr, false));
    rules_.push_back(SAMLConfig::getConfig().SecurityPolicyRuleManager.newPlugin(
        BEARER_POLICY_RULE, nullptr, false));
    policy_.reset(new SecurityPolicy());
    policy_->getRules().assign(rules_.begin(), rules_.end());
    sp_url_ = XMLString::transcode(authMetadata_->samlSpTargetUrl.c_str());
    policy_->getAudiences().push_back(sp_url_);  // NOLINT
    // read IdP metadata from file system via provider (used for authentication later)
    auto md_provider_xml =
        get_idp_metadata_provider_xml(authMetadata_->samlIdpMetadataFile);
    XercesJanitor<xercesc::DOMDocument> provider_janitor(md_provider_xml);
    try {
      idp_metadata_.reset(SAMLConfig::getConfig().MetadataProviderManager.newPlugin(
          XML_METADATA_PROVIDER, md_provider_xml->getDocumentElement(), false));
      idp_metadata_->init();
      idp_metadata_->lock();
      // read IdP metadata into XML to get it's entity ID. For some reason we can't do it
      // via idp_metadata_
      std::ifstream in(authMetadata_->samlIdpMetadataFile);
      auto md_xml = XMLToolingConfig::getConfig().getParser().parse(in);
      XercesJanitor<xercesc::DOMDocument> md_janitor(md_xml);
      XMLCh* entity_id_str = XMLString::transcode("entityID");
      const XMLCh* entity_id = md_xml->getDocumentElement()->getAttribute(entity_id_str);
      descriptor_ =
          idp_metadata_->getEntityDescriptor(MetadataProvider::Criteria(entity_id)).first;
      XMLString::release(&entity_id_str);
      RoleDescriptor* role = descriptor_->getIDPSSODescriptors().front();
      if (!descriptor_ || !role) {
        LOG(ERROR) << "Couldn't parse SAML IdP Metadata: no information about IdP";
        return;
      }
      MetadataCredentialCriteria cc(*role);
      credential_ = idp_metadata_->resolve(&cc);
      if (!credential_) {
        LOG(ERROR) << "IdP Metadata doesn't seem to provide signature credentials";
        return;
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Can not read IdP metadata file. " << e.what();
      return;
    }

    LOG(INFO) << "SAML being used for authentication. IdP: "
              << descriptor_->getEntityID();
    samlInUse_ = true;
  }
}

void SamlServer::login(std::string& username,
                       const std::string& saml_response_base64,
                       Restrictions& restrictions) {
  auto& syscat = Catalog_Namespace::SysCatalog::instance();
  std::vector<std::string> saml_roles;
  std::optional<std::string> default_db;
  if (!authenticate_user(
          username, saml_response_base64, saml_roles, restrictions, default_db)) {
    throw std::runtime_error("Invalid credentials.");
  }
  if (isRoleSyncInUse()) {
    // TODO(max): I'm not sure if there is any admin roles in SAML.
    // I suppose an administrator should manually create it as any other role
    // and manually grant everything on everything to this role.
    Catalog_Namespace::UserAlterations alts;
    alts.default_db = default_db;
    syscat.syncUserWithRemoteProvider(username, saml_roles, alts);
  }
}

static std::vector<std::string> getGroups(const saml2::Assertion* assertion) {
  std::vector<std::string> groups;
  std::unique_ptr<XMLCh, std::function<void(XMLCh*)>> attribute_name(
      XMLString::transcode("Groups"), [](XMLCh* ptr) { XMLString::release(&ptr); });
  for (const AttributeStatement* attr_stmt : assertion->getAttributeStatements()) {
    for (const Attribute* attribute : attr_stmt->getAttributes()) {
      // some saml xml responses uses a lower case Groups label.
      if (0 == XMLString::compareIString(attribute->getName(), attribute_name.get())) {
        for (const XMLObject* attribute_value : attribute->getAttributeValues()) {
          std::unique_ptr<char, std::function<void(char*)>> raw_value(
              XMLString::transcode(attribute_value->getTextContent()),
              [](char* ptr) { XMLString::release(&ptr); });
          std::vector<std::string> tmp_groups;
          // Most saml xml  has separate attributes for each group.  Some however
          // use a single field with a comma separated list.
          const std::string raw = raw_value.get();
          boost::split(tmp_groups, raw, boost::is_any_of(","));
          groups.insert(groups.end(), tmp_groups.begin(), tmp_groups.end());
        }
      }
    }
  }
  return groups;
}

static Restrictions getRestrictions(const saml2::Assertion* assertion) {
  // TODO(sy): getRestrictions: For backwards-compatibility. Maybe remove in
  // OmniSciDB 6.0. CREATE POLICY is better for managing RLS row-level security.
  std::string col_name;
  std::set<std::string> values;
  std::unique_ptr<XMLCh, std::function<void(XMLCh*)>> attribute_name(
      XMLString::transcode("Entitlement"), [](XMLCh* ptr) { XMLString::release(&ptr); });
  for (const AttributeStatement* attr_stmt : assertion->getAttributeStatements()) {
    for (const Attribute* attribute : attr_stmt->getAttributes()) {
      LOG(DEBUG4) << "Saml found attribute '" << attribute->getName() << "'";
      if (0 == XMLString::compareIString(attribute->getName(), attribute_name.get())) {
        for (const XMLObject* attribute_value : attribute->getAttributeValues()) {
          std::unique_ptr<char, std::function<void(char*)>> raw_value(
              XMLString::transcode(attribute_value->getTextContent()),
              [](char* ptr) { XMLString::release(&ptr); });
          // data should come in as {"colName":["a","b","c"]}
          const std::string raw = raw_value.get();
          using namespace rapidjson;
          Document document;
          document.Parse(raw);

          if (document.HasMember("columnName") && document["columnName"].IsString() &&
              document.HasMember("values") && document["values"].IsArray()) {
            col_name = document["columnName"].GetString();
            for (SizeType i = 0; i < document["values"].Size(); i++) {
              values.insert(document["values"][i].GetString());
            }
          } else {
            LOG(WARNING) << "SAML Entitlement '" << raw
                         << "' ignored due to incorrect format";
            continue;
          }
        }
      }
    }
  }
  Restriction restriction;
  restriction.deprecatedSamlColumnName = col_name;
  restriction.values = std::move(values);
  if (!restriction.values.empty()) {
    LOG(DEBUG1) << "Found SAML Entitlement/Restriction: " << restriction;
  }
  Restrictions restrictions;
  restrictions.emplace(restriction.getKey(), std::move(restriction));
  return restrictions;
}

static std::optional<std::string> getDefaultDB(const saml2::Assertion* assertion) {
  std::optional<std::string> default_db;
  std::unique_ptr<XMLCh, std::function<void(XMLCh*)>> attribute_name(
      XMLString::transcode("Default_DB"), [](XMLCh* ptr) { XMLString::release(&ptr); });
  for (const AttributeStatement* attr_stmt : assertion->getAttributeStatements()) {
    for (const Attribute* attribute : attr_stmt->getAttributes()) {
      LOG(DEBUG4) << "Saml found attribute '" << attribute->getName() << "'";
      if (0 == XMLString::compareIString(attribute->getName(), attribute_name.get())) {
        for (const XMLObject* attribute_value : attribute->getAttributeValues()) {
          std::unique_ptr<char, std::function<void(char*)>> raw_value(
              XMLString::transcode(attribute_value->getTextContent()),
              [](char* ptr) { XMLString::release(&ptr); });
          default_db = raw_value.get();
        }
      }
    }
  }
  if (default_db) {
    LOG(DEBUG1) << "Found Default_DB: " << *default_db;
  }
  return default_db;
}

bool SamlServer::authenticate_user(std::string& user_name,
                                   const std::string& saml_response_base64,
                                   std::vector<std::string>& saml_roles,
                                   Restrictions& restrictions,
                                   std::optional<std::string>& default_db) {
  // parse assertion from string

  std::string response_str = shared::decode_base64(saml_response_base64);
  std::stringstream ss(response_str);

  try {
    DOMDocument* response_doc = XMLToolingConfig::getConfig().getParser().parse(ss);
    XercesJanitor<DOMDocument> response_janitor(response_doc);

    std::unique_ptr<opensaml::saml2p::Response> response(
        dynamic_cast<opensaml::saml2p::Response*>(
            XMLObjectBuilder::getBuilder(response_doc->getDocumentElement())
                ->buildFromDocument(response_doc)));
    response_janitor.release();

    if (response->getAssertions().empty()) {
      LOG(WARNING) << "Error validating SAML response. SAML response doesn't contain any "
                      "assertions";
      return false;
    }
    const auto* assertion = response->getAssertions().front();

    // check that issuer and entity in metadata match
    if (!XMLString::equals(assertion->getIssuer()->getTextContent(),
                           descriptor_->getEntityID())) {
      LOG(WARNING) << "SAML assertion comes from an untrusted IdP:"
                   << "\nExpected: " << descriptor_->getEntityID()
                   << "\nGot: " << response->getIssuer();
      return false;
    }
    try {
      // check signatures for Response and/or Assertion, if configured
      SignatureProfileValidator spv;
      xmlsignature::SignatureValidator sv(credential_);
      if (authMetadata_->samlSignedAssertion) {
        if (auto* signature = assertion->getSignature()) {
          spv.validateSignature(*signature);
          sv.validate(signature);
        } else {
          LOG(WARNING) << "Error validationg SAML response: assertions should be signed";
          return false;
        }
      }
      if (authMetadata_->samlSignedResponse) {
        if (auto* signature = response->getSignature()) {
          spv.validateSignature(*signature);
          sv.validate(signature);
        } else {
          LOG(WARNING) << "Error validationg SAML response: it should be signed";
          return false;
        }
      }
      if (auto* signature = response->getSignature()) {
        spv.validateSignature(*signature);
        sv.validate(signature);
      }
    } catch (const XMLToolingException& e) {
      LOG(WARNING) << "Error validationg SAML response. " << e.what();
      return false;
    }

    // validate that the assertion is not outdated and is issued for us
    // NOTE(max): I'm not sure why it's necessary. CONDITIONS_POLICY_RULE is supposed to
    // take care of dates. But it does so with delay (on my tests about 3 minutes).
    SubjectConfirmationData* data =
        dynamic_cast<SubjectConfirmationData*>(assertion->getSubject()
                                                   ->getSubjectConfirmations()
                                                   .front()
                                                   ->getSubjectConfirmationData());
    if (data && time(nullptr) > data->getNotOnOrAfterEpoch()) {
      LOG(WARNING) << "Error validationg SAML response. It is no longer valid";
      return false;
    }

    {
      heavyai::unique_lock<heavyai::shared_mutex> policy_lock(policy_mutex_);
      policy_->reset();
      policy_->setTime(time(nullptr));
      policy_->evaluate(*assertion);
    }

    char* name = XMLString::transcode(assertion->getSubject()->getNameID()->getName());
    user_name = name;
    XMLString::release(&name);
    saml_roles = getGroups(assertion);
    restrictions = getRestrictions(assertion);
    default_db = getDefaultDB(assertion);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Error validating SAML response. " << e.what();
    return false;
  }

  std::string const loggable =
      Catalog_Namespace::g_log_user_id ? std::string("") : user_name + ' ';
  LOG(INFO) << " User " << loggable << "connecting with SAML authentication";
  return true;
}

bool SamlServer::inUse() const {
  return samlInUse_;
}

bool SamlServer::isRoleSyncInUse() const {
  return inUse() && authMetadata_->samlSyncRoles;
}

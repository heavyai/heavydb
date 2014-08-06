/**
 * @file	Parse.h
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <string>
 
// forward declarations
namespace SQL_Namespace { class ASTNode; }
namespace RA_Namespace { class RelAlgNode; }

namespace Parse_Namespace {

/**
 * @brief Returns the root of the SQL abstract syntax tree parsed from the input string.
 */
SQL_Namespace::ASTNode* parse_sql(const std::string &input, std::string &errMsg);

/**
 * @brief Returns the root of the RA abstract syntax tree parsed from the input string.
 */
RA_Namespace::RelAlgNode* parse_ra(const std::string &input, std::string &errMsg);

} // Parse_Namespace

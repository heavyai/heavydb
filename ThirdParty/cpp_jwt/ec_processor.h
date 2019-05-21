#ifndef EC_PROCESSOR_H
#define EC_PROCESSOR_H
#include <string>

bool ec_verify(const std::string& payload,
               const std::string& signature,
               const std::string& ecPubKey);

#endif

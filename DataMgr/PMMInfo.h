
#pragma once

#include <string>

struct PMMInfo {
  bool pmm;
  std::string pmm_path;
  bool pmm_store;
  std::string pmm_store_path;

  static PMMInfo disabled() { return PMMInfo{false, "", false, ""}; }
};
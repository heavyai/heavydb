/*
 * Copyright 2020 OmniSci, Inc.
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

#include "OSDependent/omnisci_hostname.h"

#include <windows.h>

namespace omnisci {
std::string get_hostname() {
  static constexpr DWORD kSize = MAX_COMPUTERNAME_LENGTH + 1;
  DWORD buffer_size = kSize;
  char hostname[MAX_COMPUTERNAME_LENGTH + 1];
  if (GetComputerNameA(hostname, &buffer_size)) {
    return {hostname};
  } else {
    return {};
  }
}
}  // namespace omnisci

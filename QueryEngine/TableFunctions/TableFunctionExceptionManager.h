/*
 * Copyright 2021 OmniSci, Inc.
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

#include <string>

#include "Logger/Logger.h"

/*
  The TableFunctionExceptionManager allows table functions to
  communicate error/exception messages up to the execution
  context. Table functions can return with a call to
  `table_function_error` with an error message. This will indicate
  to the execution context that an error ocurred within the table
  function, and the error will be propagated as an exception.
*/

// Use a set negative value to distinguish from already-existing
// negative return values
enum TableFunctionError : int32_t {
  GenericError = -0x75BCD15,
};

struct TableFunctionExceptionManager {
  /*
  TableFunctionExceptionManager is a dynamic singleton:
  `get_singleton()` returns a pointer to the singleton instance when
  in the scope of TableFunctionExceptionManager life-time,
  otherwise returns nullptr. For internal usage.
  */
  static TableFunctionExceptionManager*& get_singleton() {
    static TableFunctionExceptionManager* instance = nullptr;
    return instance;
  }

  TableFunctionExceptionManager() {
    set_singleton(this);  // start of singleton life
  }

  ~TableFunctionExceptionManager() {
    set_singleton(nullptr);  // end of singleton life
  }

  const char* get_error_message() const { return error_message_.c_str(); }

  void set_error_message(const char* msg) { error_message_ = std::string(msg); }

 private:
  std::string error_message_;

  static void set_singleton(TableFunctionExceptionManager* instance) {
    auto& instance_ = get_singleton();
    // ensure being singleton
    if (instance) {
      CHECK(instance_ == nullptr);
    } else {
      CHECK(instance_ != nullptr);
    }
    instance_ = instance;
  }
};

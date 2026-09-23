/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file   SinkedPtr.h
 * @author Steve Blackmon <steve.blackmon@omnisci.com>
 * @brief  Unique and shared pointer wrapper that requires using a sink
 * @description Unique Graphics resources require complex lifetime management that
 *              goes beyond standard memory objects, including host / device
 *              synchronization. This requires creation and destruction to be
 *              carefully managed at a system level. To help prevent leaks,
 *              leverage unique_ptr with a custom deleter. The deleter uses a private
 *              "cookie" only accessible by the SINK, which can "unlock" the pointer
 *              to allow it to be deleted.
 *
 *              The sinked_ptr does not own nor free the resource, it acts as a
 *              check to ensure the proper sink function is called, it is up to
 *              the SINK object to ensure the resources are freed
 *              Failure to use the proper sink function will cause trigger a CHECK
 *
 *              Along with the Deleter functor, there is an alias template for
 *              unique_ptr, along with a set of make_sinked_ptr
 *              helper functions to simplify construction and improve exception
 *              safety.
 */
#pragma once

#include <memory>
#include <type_traits>

#ifndef __CUDACC__
#include <string_view>
#endif

#include "Logger/Logger.h"

namespace gfx {

#ifndef __CUDACC__
namespace detail {

// Slices the type name out of the compiler's own signature for this function,
// which unlike typeid() also works for types that are only forward-declared.
// The template parameter is named U so that the "U = " marker is unambiguous.
template <typename U>
constexpr std::string_view sinked_type_name() {
  constexpr std::string_view signature{__PRETTY_FUNCTION__};
  constexpr std::string_view marker{"U = "};
  constexpr auto marker_pos = signature.find(marker);
  static_assert(marker_pos != std::string_view::npos, "unsupported compiler");
  constexpr auto begin = marker_pos + marker.size();
  // gcc appends further template substitutions after a ';', clang ends at the ']'
  constexpr auto end = signature.find_first_of(";]", begin);
  static_assert(end != std::string_view::npos, "unsupported compiler");
  return signature.substr(begin, end - begin);
}

}  // namespace detail
#endif

template <typename T, typename SINK>
struct SinkedDeleter {
  SinkedDeleter() = default;
  void operator()(T* r) {
    // The deleter must be default constructible for sinked_ptr to work with
    // standard containers
    static_assert(std::is_default_constructible<SinkedDeleter<T*, SINK>>::value,
                  "SinkedDeleter must be default constructible");

#ifndef __CUDACC__
    LOG_IF(ERROR, allow_delete_ == false)
        << " sinked_ptr leak! " << detail::sinked_type_name<T>()
        << " must be destroyed using " << detail::sinked_type_name<SINK>() << " sink";
#else
    CHECK(allow_delete_);
#endif
  }

 private:
  bool allow_delete_{false};
  friend SINK;
};

template <typename T, typename SINK>
using sinked_ptr = std::unique_ptr<T, SinkedDeleter<T, SINK>>;

//
// make_sinked_ptr helpers
//

// Wrap a constructed type in a sinked_ptr of the same type (will deduce
// T from the passed pointer's type)
template <typename T, typename SINK>
inline auto make_sinked_ptr(T* t) {
  SinkedDeleter<T, SINK> del;
  return sinked_ptr<T, SINK>(t, del);
}

// Wrap a constructed type of BASE type in sinked_ptr of T type with
// automatic static_casting. Requires explicit T type declaration
template <typename T, typename BASE, typename SINK>
inline auto make_sinked_ptr(BASE* r) {
  SinkedDeleter<T, SINK> del;
  return sinked_ptr<T, SINK>(static_cast<T*>(r), del);
}

}  // namespace gfx

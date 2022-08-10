/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include <optional>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/optional.hpp>

namespace boost {
namespace serialization {

/**
 * boost surprisingly does not have native support for serializing std::optional values
 * from STL. This is a workaround for that by converting a std::optional to a
 * boost::optional for serialization. There is a compilation issue with using the standard
 * boost::serialization::split_free to handle different pathways for load/store of a
 * std::optional. This compilation issue requires a workaround. The commented out code
 * below is what would be used if the split_free route compiles, so keeping it around in
 * case one day it does work as it would be preferred. More on the compilation issue in
 * comments below.
 */

// NOTE: uncomment if we find that a more recent gcc or boost works. For more, see
// comments below

// template <class Archive, typename T> void save(Archive& ar, const
// std::optional<T>& in_opt, const unsigned int version) {
//   boost::optional<T> boost_opt;
//   if (in_opt) {
//     boost_opt = *in_opt;
//   }
//   ar& boost_opt;
// }

// NOTE: uncomment if we find that a more recent gcc or boost works. For more, see
// comments below.

// template <class Archive, typename T>
// void load(Archive& ar, std::optional<T>& out_opt, const unsigned int version) {
//   boost::optional<T> boost_opt;
//   ar& boost_opt;

//   if (boost_opt) {
//     out_opt = *boost_opt;
//   } else {
//     // probably unnecessary
//     out_opt.reset();
//   }
// }

template <class Archive, class T>
void serialize(Archive& ar, std::optional<T>& in_opt, const unsigned int version) {
  // NOTE: uncomment if we find that a more recent gcc or boost works
  // ::boost::serialization::split_free(ar, in_opt, version);

  // The if constexpr block below is a workaround to what appears to be a gcc v9.4.0 bug.
  // The preferred way to split the load/save code paths is to use the
  // boost::serialization::split_free:
  // https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#splitting
  //
  // However, doing it the split_free way, which means overriding the save/load free
  // functions for std::optional, results in a very strange compilation error.
  // The parent header file for doing RelAlgDag serialization does a:
  // #include <boost/serialization/shared_ptr.hpp>
  //
  // shared_ptr.hpp in turn does a:
  // #include <boost/serialization/shared_ptr_helper.hpp>
  // see:
  // https://www.boost.org/doc/libs/1_74_0/boost/serialization/shared_ptr.hpp)
  //
  // shared_ptr_helper.hpp has this ifndef block:
  //
  // #ifndef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
  // template<class Archive, template<class U> class SPT >
  // void load(
  //   Archive & ar,
  //   SPT< class U > &t,
  //   const unsigned int file_version
  // );
  // #endif
  //
  // see: https://www.boost.org/doc/libs/1_74_0/boost/serialization/shared_ptr_helper.hpp
  //
  // That load free-function declaration confuses gcc, and you get a compilation error
  // like this:
  //
  // clang-format off
  //
  // In file included from /usr/include/c++/9/bits/move.h:55,
  //                  from /usr/include/c++/9/bits/stl_pair.h:59,
  //                  from /usr/include/c++/9/utility:70,
  //                  from /usr/include/c++/9/algorithm:60,
  //                  from ../QueryEngine/Execute.h:20,
  //                  from heavydb-internal/build/QueryEngine/CMakeFiles/QueryEngine.dir/cmake_pch.hxx:5,
  //                  from <command-line>:
  // /usr/include/c++/9/type_traits: In instantiation of ‘struct std::__is_trivially_copy_constructible_impl<boost::serialization::U, true>’:
  // /usr/include/c++/9/type_traits:1157:12:   required from ‘struct std::is_trivially_copy_constructible<boost::serialization::U>’
  // /usr/include/c++/9/type_traits:2938:25:   required from ‘constexpr const bool std::is_trivially_copy_constructible_v<boost::serialization::U>’
  // /usr/include/c++/9/optional:469:12:   required by substitution of ‘template<class Archive, template<class U> class SPT> void boost::serialization::load(Archive&, SPT<boost::serialization::U>&, unsigned int) [with Archive = boost::archive::text_iarchive; SPT = <missing>]’
  // /usr/include/boost/serialization/split_free.hpp:58:13:   required from ‘static void boost::serialization::free_loader<Archive, T>::invoke(Archive&, T&, unsigned int) [with Archive = boost::archive::text_iarchive; T = std::optional<long unsigned int>]’
  // /usr/include/boost/serialization/split_free.hpp:74:18:   required from ‘void boost::serialization::split_free(Archive&, T&, unsigned int) [with Archive = boost::archive::text_iarchive; T = std::optional<long unsigned int>]’
  // ../QueryEngine/RelAlgSerializerOptional.h:52:37:   [ skipping 42 instantiation contexts, use -ftemplate-backtrace-limit=0 to disable ]
  // /usr/include/boost/archive/detail/iserializer.hpp:624:18:   required from ‘void boost::archive::load(Archive&, T&) [with Archive = boost::archive::text_iarchive; T = RelAlgDag]’
  // /usr/include/boost/archive/detail/common_iarchive.hpp:67:22:   required from ‘void boost::archive::detail::common_iarchive<Archive>::load_override(T&) [with T = RelAlgDag; Archive = boost::archive::text_iarchive]’
  // /usr/include/boost/archive/basic_text_iarchive.hpp:70:9:   required from ‘void boost::archive::basic_text_iarchive<Archive>::load_override(T&) [with T = RelAlgDag; Archive = boost::archive::text_iarchive]’
  // /usr/include/boost/archive/text_iarchive.hpp:82:52:   required from ‘void boost::archive::text_iarchive_impl<Archive>::load_override(T&) [with T = RelAlgDag; Archive = boost::archive::text_iarchive]’
  // /usr/include/boost/archive/detail/interface_iarchive.hpp:68:9:   required from ‘Archive& boost::archive::detail::interface_iarchive<Archive>::operator>>(T&) [with T = RelAlgDag; Archive = boost::archive::text_iarchive]’
  // ../QueryEngine/RelAlgDagSerializer/RelAlgDagSerializer.cpp:70:12:   required from here
  // /usr/include/c++/9/type_traits:1150:12: error: invalid use of incomplete type ‘class boost::serialization::U’
  //  1150 |     struct __is_trivially_copy_constructible_impl<_Tp, true>
  //       |            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // In file included from /usr/include/boost/serialization/shared_ptr.hpp:29,
  //                  from ../QueryEngine/RelAlgDagSerializer/RelAlgDagSerializer.h:25,
  //                  from ../QueryEngine/RelAlgDagSerializer/RelAlgDagSerializer.cpp:17:
  // /usr/include/boost/serialization/shared_ptr_helper.hpp:45:16: note: forward declaration of ‘class boost::serialization::U’
  //    45 |     SPT< class U > &t,
  //       |                ^
  // In file included from /usr/include/c++/9/bits/move.h:55,
  //                  from /usr/include/c++/9/bits/stl_pair.h:59,
  //                  from /usr/include/c++/9/utility:70,
  //                  from /usr/include/c++/9/algorithm:60,
  //                  from ../QueryEngine/Execute.h:20,
  //                  from heavydb-internal/build/QueryEngine/CMakeFiles/QueryEngine.dir/cmake_pch.hxx:5,
  //                  from <command-line>:
  //
  //
  // clang-format on
  //
  // It appears as tho there's a gcc bug that confuses the load() free function
  // declaration in shared_ptr_helper.hpp when trying to resolve overrides with the
  // two-argument template and somehow substitutes the second template argument with a
  // boost::serialization::U class type, which does not exist.
  //
  // So the workaround is to use the if contexpr switch and pull the definitions of the
  // load/save free functions for std::optional directly into the serialize override,
  // which works fine and has no other conflicts.
  //
  // If we find that we upgrade to a more recent version of gcc and it works, we can move
  // back to the split_free approach, which is commented out above.

  if constexpr (std::is_same_v<::boost::archive::text_iarchive, Archive>) {
    // load case
    boost::optional<T> boost_opt;
    (ar & boost_opt);

    if (boost_opt) {
      in_opt = *boost_opt;
    } else {
      // probably unnecessary
      in_opt.reset();
    }
  } else {
    // save case
    static_assert(std::is_same_v<::boost::archive::text_oarchive, Archive>);
    boost::optional<T> boost_opt;
    if (in_opt) {
      boost_opt = *in_opt;
    }
    (ar & boost_opt);
  }
}

}  // namespace serialization
}  // namespace boost

#ifndef CONSTEXPRLIB_H
#define CONSTEXPRLIB_H

namespace Experimental {

template <typename T1, typename T2>
struct ConstExprPair {
  using first_type = T1;
  using second_type = T2;

  constexpr ConstExprPair() : first(), second() {}
  constexpr ConstExprPair(first_type const& t1, second_type const& t2)
      : first(t1), second(t2) {}

  template <typename U1, typename U2>
  constexpr ConstExprPair(ConstExprPair<U1, U2> const& p)
      : first(p.first), second(p.second) {}

  constexpr ConstExprPair& operator=(ConstExprPair& other) {
    first = other.first;
    second = other.second;
    return *this;
  }

  constexpr void swap(ConstExprPair& other) noexcept {
    first_type tempt1 = first;
    second_type tempt2 = second;
    first = other.first;
    second = other.second;
    other.first = tempt1;
    other.second = tempt2;
  }

  first_type first;
  second_type second;
};

}  // namespace Experimental

#endif

#ifndef SQLTYPEUTILITIES_H
#define SQLTYPEUTILITIES_H

template <SQLTypes... TYPE_SET>
class OnTypesetMember {
 public:
  OnTypesetMember() = delete;

  template <typename SQL_INFO_TYPE>
  OnTypesetMember(SQL_INFO_TYPE const& s)
      : resolved_(internalResolveType<SQL_INFO_TYPE, TYPE_SET...>(s)) {}

  template <typename SQL_INFO_TYPE, typename FUNCTOR_TYPE>
  OnTypesetMember(SQL_INFO_TYPE const& s, FUNCTOR_TYPE f)
      : resolved_(internalResolveType<SQL_INFO_TYPE, TYPE_SET...>(s)) {
    if (resolved_)
      f();
  }

  template <typename SQL_INFO_TYPE,
            typename SUCCESS_FUNCTOR_TYPE,
            typename FAILURE_FUNCTOR_TYPE>
  OnTypesetMember(SQL_INFO_TYPE const& s,
                  SUCCESS_FUNCTOR_TYPE success,
                  FAILURE_FUNCTOR_TYPE failure)
      : resolved_(internalResolveType<SQL_INFO_TYPE, TYPE_SET...>(s)) {
    resolved_ ? success() : failure();
  }

  template <typename SQL_INFO_TYPE, typename FUNCTOR_TYPE>
  bool operator()(SQL_INFO_TYPE const& s, FUNCTOR_TYPE failure) const {
    bool resolved = internalResolveType<TYPE_SET...>(s);
    if (resolved)
      failure();
    return resolved;
  }

  template <typename SQL_INFO_TYPE,
            typename SUCCESS_FUNCTOR_TYPE,
            typename FAILURE_FUNCTOR_TYPE>
  bool operator()(SQL_INFO_TYPE const& s,
                  SUCCESS_FUNCTOR_TYPE success,
                  FAILURE_FUNCTOR_TYPE failure) const {
    bool resolved = internalResolveType<SQL_INFO_TYPE, TYPE_SET...>(s);
    resolved ? success() : failure();
    return resolved;
  }

  operator bool() const { return resolved_; }

 private:
  template <typename SQL_INFO_TYPE, SQLTypes TYPE>
  bool internalResolveType(SQL_INFO_TYPE const& s) const {
    if (s.get_type() == TYPE)
      return true;
    return false;
  }

  template <typename SQL_INFO_TYPE,
            SQLTypes TYPE,
            SQLTypes SECOND_TYPE,
            SQLTypes... REMAINING_TYPES>
  bool internalResolveType(SQL_INFO_TYPE const& s) const {
    if (s.get_type() == TYPE)
      return true;
    return internalResolveType<SQL_INFO_TYPE, SECOND_TYPE, REMAINING_TYPES...>(s);
  }

  bool const resolved_ = false;
};

template <SQLTypes... TYPE_SET, typename SQL_INFO_TYPE>
bool is_member_of_typeset(SQL_INFO_TYPE const& s) {
  return OnTypesetMember<TYPE_SET...>(s);
}

template <SQLTypes... TYPE_SET, typename SUCCESS_FUNCTOR_TYPE, typename SQL_INFO_TYPE>
bool on_member_of_typeset(SQL_INFO_TYPE const& s, SUCCESS_FUNCTOR_TYPE success) {
  return OnTypesetMember<TYPE_SET...>(s, success);
}

template <SQLTypes... TYPE_SET,
          typename SUCCESS_FUNCTOR_TYPE,
          typename FAILURE_FUNCTOR_TYPE,
          typename SQL_INFO_TYPE>
bool on_member_of_typeset(SQL_INFO_TYPE const& s,
                          SUCCESS_FUNCTOR_TYPE success,
                          FAILURE_FUNCTOR_TYPE failure) {
  return OnTypesetMember<TYPE_SET...>(s, success, failure);
}

#endif

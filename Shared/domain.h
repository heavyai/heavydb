/**
 * @file		domain.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Definitions for domain analysis on expressions
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef DOMAIN_H
#define DOMAIN_H

#include <vector>
#include <boost/variant.hpp>
#include "../Shared/sqltypes.h"

enum DomainKind {
  kDOMAIN_UNKNOWN,
  kDOMAIN_RANGE,
  kDOMAIN_SET,
  kDOMAIN_COLUMN,
  kDOMAIN_UNION,
  kDOMAIN_INTERSECT
};

struct DomainRange {
  Datum domain_min;
  Datum domain_max;
  bool domain_nullable;
};

typedef std::vector<Datum> DomainSet;

struct DomainColumn {
  int table_id;
  int column_id;
};

class Domain;

typedef std::vector<Domain> DomainChildren;

class Domain {
  public:
    Domain(const SQLTypeInfo &t) : domain_kind(kDOMAIN_UNKNOWN), domain_type(t) {}
    Domain(const SQLTypeInfo &t, const DomainRange &r) : domain_kind(kDOMAIN_RANGE), domain_type(t), domain({r}) {};
    Domain(const SQLTypeInfo &t, const DomainSet &s) : domain_kind(kDOMAIN_SET), domain_type(t), domain(s) {};
    Domain(const SQLTypeInfo &t, const DomainColumn &c) : domain_kind(kDOMAIN_COLUMN), domain_type(t), domain(c) {}
    Domain(const SQLTypeInfo &t, const DomainKind k, const DomainChildren &ch) : domain_kind(k), domain_type(t), domain(ch) {}
    const DomainKind get_domain_kind() const { return domain_kind; }
    const SQLTypeInfo &get_domain_type() const { return domain_type; }
    const DomainRange &get_domain_range() const { return *boost::get<DomainRange>(&domain); }
    const DomainSet &get_domain_set() const { return *boost::get<DomainSet>(&domain); }
    const DomainColumn &get_domain_column() const { return *boost::get<DomainColumn>(&domain); }
    const DomainChildren &get_domain_children() const { return *boost::get<DomainChildren>(&domain); }
  private:
    DomainKind domain_kind;
    SQLTypeInfo domain_type;
    boost::variant<DomainRange, DomainSet, DomainColumn, DomainChildren> domain;
};

#endif // DOMAIN_H

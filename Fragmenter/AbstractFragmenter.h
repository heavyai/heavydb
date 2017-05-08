/*
 * Copyright 2017 MapD Technologies, Inc.
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

/**
 * @file    AbstractFragmenter.h
 * @author  Todd Mostak <todd@map-d.com
 */

#ifndef _ABSTRACT_FRAGMENTER_H
#define _ABSTRACT_FRAGMENTER_H

#include "../Shared/sqltypes.h"
#include "Fragmenter.h"
#include <vector>
#include <string>

// Should the ColumnInfo and FragmentInfo structs be in
// AbstractFragmenter?

namespace Data_Namespace {
class AbstractBuffer;
class AbstractDataMgr;
};

namespace Fragmenter_Namespace {

/*
 * @type AbstractFragmenter
 * @brief abstract base class for all table partitioners
 *
 * The virtual methods of this class provide an interface
 * for an interface for getting the id and type of a
 * partitioner, inserting data into a partitioner, and
 * getting the partitions (fragments) managed by a
 * partitioner that must be queried given a predicate
 */

class AbstractFragmenter {
 public:
  virtual ~AbstractFragmenter() {}

  /**
   * @brief Should get the partitions(fragments)
   * where at least one tuple could satisfy the
   * (optional) provided predicate, given any
   * statistics on data distribution the partitioner
   * keeps. May also prune the predicate.
   */

  // virtual void getFragmentsForQuery(QueryInfo &queryInfo, const void *predicate = 0) = 0;
  virtual TableInfo getFragmentsForQuery() = 0;

  /**
   * @brief Given data wrapped in an InsertData struct,
   * inserts it into the correct partitions
   * with locks and checkpoints
   */

  virtual void insertData(const InsertData& insertDataStruct) = 0;

  /**
   * @brief Given data wrapped in an InsertData struct,
   * inserts it into the correct partitions
   * No locks and checkpoints taken needs to be managed externally
   */

  virtual void insertDataNoCheckpoint(const InsertData& insertDataStruct) = 0;

  /**
   * @brief Will truncate table to less than maxRows by dropping
   * fragments
   */

  virtual void dropFragmentsToSize(const size_t maxRows) = 0;

  /**
   * @brief Gets the id of the partitioner
   */
  virtual int getFragmenterId() = 0;

  /**
   * @brief Gets the string type of the partitioner
   * @todo have a method returning the enum type?
   */

  virtual std::string getFragmenterType() = 0;
};

}  // Fragmenter_Namespace

#endif  // _ABSTRACT_FRAGMENTER_H

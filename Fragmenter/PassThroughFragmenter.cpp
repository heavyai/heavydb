/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "Fragmenter/PassThroughFragmenter.h"

namespace Fragmenter_Namespace {

void PassThroughFragmenter::insertChunksImpl(const InsertChunks& insert_chunks) {
  auto delete_column_id = findDeleteColumnId();

  // verify that all chunks to be inserted have same number of rows, otherwise the input
  // data is malformed
  auto num_rows = validateSameNumRows(insert_chunks);

  // PassThroughFragmenter assumes that any call to insertChunksImpl will fit into a
  // single fragment, so we don't have to create more than one fragment or loop over the
  // data.  Every call simply creates a new fragment and inserts the data.
  CHECK_GE(maxFragmentRows_, num_rows.value());

  auto valid_row_indices = insert_chunks.valid_row_indices;
  size_t num_rows_left = valid_row_indices.size();
  size_t num_rows_inserted = 0;
  const size_t num_rows_to_insert = num_rows_left;

  if (num_rows_left == 0) {
    return;
  }

  auto new_fragment = createNewFragment(defaultInsertLevel_);
  CHECK(new_fragment);
  size_t fragment_idx = fragmentInfoVec_.size() - 1;

  insertChunksIntoFragment(insert_chunks,
                           delete_column_id,
                           new_fragment,
                           num_rows_to_insert,
                           num_rows_inserted,
                           num_rows_left,
                           valid_row_indices,
                           fragment_idx);

  CHECK_EQ(num_rows_left, 0U);

  numTuples_ += num_rows_inserted;
  dropFragmentsToSizeNoInsertLock(maxRows_);
}

}  // namespace Fragmenter_Namespace

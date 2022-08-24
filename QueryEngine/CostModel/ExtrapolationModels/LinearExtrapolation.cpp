/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "LinearExtrapolation.h"

#include <algorithm>

namespace costmodel {

size_t LinearExtrapolation::getExtrapolatedData(size_t bytes) {
  size_t id1, id2;
  Detail::Measurement tmp = {.bytes = bytes, .milliseconds = 0};

  auto iter =
      std::upper_bound(measurement.begin(), measurement.end(), tmp, Detail::BytesOrder());

  if (iter == measurement.begin()) {
    id1 = 0;
    id2 = 1;
  } else if (iter == measurement.end()) {
    id1 = measurement.size() - 2;
    id2 = measurement.size() - 1;
  } else {
    id2 = iter - measurement.begin();
    id1 = id2 - 1;
  }

  size_t y1 = measurement[id1].milliseconds, y2 = measurement[id2].milliseconds;
  size_t x1 = measurement[id1].bytes, x2 = measurement[id2].bytes;

  return y1 + ((double)bytes - x1) / (x2 - x1) * (y2 - y1);
}

}  // namespace costmodel

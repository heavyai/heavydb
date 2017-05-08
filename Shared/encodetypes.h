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

#ifndef ENCODETYPES_H
#define ENCODETYPES_H

enum EncodingAlgo { NONE = 0, FIXED = 1, DIFFERENTIAL = 2 };

enum EncodingType { kINT8 = 0, kINT16 = 1, kINT32 = 2, kINT64 = 3, kUINT8 = 4, kUINT16 = 5, kUINT32 = 6, kUINT64 = 7 };

#endif  // ENCODETYPES_H

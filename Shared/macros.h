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

#define PRINT_SLINE(n)        \
  for (int i = 0; i < n; ++i) \
    printf("-");              \
  printf("\n");
#define PRINT_DLINE(n)        \
  for (int i = 0; i < n; ++i) \
    printf("=");              \
  printf("\n");
#define PRINT_DEBUG(msg) printf("[%s:%d] %s\n", __func__, __LINE__, msg);

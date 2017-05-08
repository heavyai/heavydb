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

/*
 * File:   testing.h
 * Author: sstewart
 *
 * Created on June 21, 2014, 2:33 PM
 */

#ifndef TESTING_H
#define TESTING_H

namespace Testing {

static unsigned pass = 0;
static unsigned fail = 0;
static unsigned skip = 0;

#define PPASS(msg) \
  (printf("[%s:%d] %s PASS(%d):\t%s%s\n", __FILE__, __LINE__, ANSI_COLOR_GREEN, ++pass, msg, ANSI_COLOR_RESET))
#define PFAIL(msg) \
  (printf("[%s:%d] %s FAIL(%d):\t%s%s\n", __FILE__, __LINE__, ANSI_COLOR_RED, ++fail, msg, ANSI_COLOR_RESET))
#define PSKIP(msg) \
  (printf("[%s:%d] %s SKIP(%d):\t%s%s\n", __FILE__, __LINE__, ANSI_COLOR_BLUE, ++skip, msg, ANSI_COLOR_RESET))
#define PCLEAR \
  pass = 0;    \
  fail = 0;    \
  skip = 0;

void printTestSummary() {
  printf("pass=%u fail=%u skip=%u total=%u (%.1f%% %.1f%% %.1f%%)\n",
         pass,
         fail,
         skip,
         (pass + fail),
         ((double)pass / (pass + fail)) * 100,
         ((double)fail / (pass + fail)) * 100,
         ((double)skip / (pass + fail)) * 100);
}
}
#endif /* TESTING_H */

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

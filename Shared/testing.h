/* 
 * File:   testing.h
 * Author: sstewart
 *
 * Created on June 21, 2014, 2:33 PM
 */

#ifndef TESTING_H
#define	TESTING_H

namespace Testing {

static unsigned pass = 0;
static unsigned fail = 0;
static unsigned skip = 0;

#define PPASS(msg) (printf("%s: %s[%d] PASS(%d):\t%s%s\n", __FILE__, ANSI_COLOR_GREEN, (pass+fail), ++pass, msg, ANSI_COLOR_RESET))
#define PFAIL(msg) (printf("%s: %s[%d] FAIL(%d):\t%s%s\n", __FILE__, ANSI_COLOR_RED, (pass+fail), ++fail, msg, ANSI_COLOR_RESET))
#define PSKIP(msg) (printf("%s: %s[%d] SKIP(%d):\t%s%s\n", __FILE__, ANSI_COLOR_BLUE, (pass+fail), ++skip, msg, ANSI_COLOR_RESET))

void printTestSummary() {
    printf("pass=%u fail=%u skip=%u total=%u (%.1f%% %.1f%% %.1f%%)\n", 
            pass, fail, skip, (pass+fail), ((double)pass/(pass+fail))*100, ((double)fail/(pass+fail))*100, ((double)skip/(pass+fail))*100);
}

}
#endif	/* TESTING_H */


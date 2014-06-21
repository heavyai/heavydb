/* 
 * File:   testing.h
 * Author: sstewart
 *
 * Created on June 21, 2014, 2:33 PM
 */

#ifndef TESTING_H
#define	TESTING_H

unsigned pass = 0;
unsigned fail = 0;
unsigned skip = 0;
unsigned total = 0;
#define PPASS(msg) (printf("%s: %s[%d] PASS(%d):\t%s%s\n", __FILE__, ANSI_COLOR_GREEN, ++total, ++pass, msg, ANSI_COLOR_RESET))
#define PFAIL(msg) (printf("%s: %s[%d] FAIL(%d):\t%s%s\n", __FILE__, ANSI_COLOR_RED, ++total, ++fail, msg, ANSI_COLOR_RESET))
#define PSKIP(msg) (printf("%s: %s[%d] SKIP(%d):\t%s%s\n", __FILE__, ANSI_COLOR_BLUE, ++total, ++skip, msg, ANSI_COLOR_RESET))

void printTestSummary() {
    printf("pass=%u fail=%u skip=%u total=%u (%.1f%% %.1f%% %.1f%%)\n", 
            pass, fail, skip, total, ((double)pass/total)*100, ((double)fail/total)*100, ((double)skip/total)*100);
}


#endif	/* TESTING_H */


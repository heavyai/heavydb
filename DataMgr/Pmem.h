#ifdef HAVE_DCPMM

#ifndef PMEM_H
#define PMEM_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define PmemClflushopt(addr) \
	asm volatile(".byte 0x66; clflush %0" : "+m" \
		(*(volatile char *)(addr)));
#define PmemClwb(addr)\
	asm volatile(".byte 0x66; xsaveopt %0" : "+m" \
		(*(volatile char *)(addr)));

#define PmemFence	_mm_sfence

#define CACHELINE_SIZE	64

extern "C" {
	void PmemFlush(const void *addr, size_t len);
	void PmemPersist(const void *addr, size_t len);
	void PmemMemCpy(char *dest, const char *src, const size_t len);
	void PmemMemSet(char *dest, int c, size_t len);
}

#endif /* PMEM_H */

#endif /* HAVE_DCPMM */

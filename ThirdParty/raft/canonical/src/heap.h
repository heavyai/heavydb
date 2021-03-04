/* Internal heap APIs. */

#ifndef HEAP_H_
#define HEAP_H_

#include <stddef.h>

void *HeapMalloc(size_t size);

void *HeapCalloc(size_t nmemb, size_t size);

void *HeapRealloc(void *ptr, size_t size);

void HeapFree(void *ptr);

#endif /* HEAP_H_ */

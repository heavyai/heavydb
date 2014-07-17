//
#include <cstring>
#include <cassert>
#include "Buffer.h"

namespace Buffer_Namespace {

Buffer::Buffer(mapd_addr_t *host_ptr, mapd_size_t numPages, mapd_size_t pageSize) {
	assert(pageSize > 0);
	host_ptr_ = host_ptr;
    length_ = 0;
    pageSize_ = pageSize;
    pins_ = 1;
    dirty_ = false;
    for (int i = 0; i < numPages; ++i)
        pages_.push_back(new Page(host_ptr + (i * pageSize), false));
}

Buffer::~Buffer() {
	while (pages_.size() > 0) {
		delete pages_.back();
		pages_.pop_back();
	}
}

bool Buffer::write(mapd_size_t offset, mapd_size_t n, mapd_addr_t *src) {
	assert(n > 0 && src);

	// check for buffer overflow
	mapd_size_t bufSize = size();
	if (length_ + n > bufSize)
		return false;
	
	// write source contents to buffer
	memcpy(host_ptr_ + offset, src, n);
	length_ += n;

	// update dirty flags
	dirty_ = true;
	mapd_size_t firstPage = offset / pageSize_;
	mapd_size_t lastPage = (offset + n) / pageSize_;
	for (mapd_size_t i = firstPage; i <= lastPage; ++i)
		pages_[i]->dirty = true;

	return true;
}

bool Buffer::append(mapd_size_t n, mapd_addr_t *src) {
	assert(n > 0 && src);
	return write(length_, n, src);
}
   
void Buffer::copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t *dest) {
	assert(n > 0 && dest);;
	memcpy(dest, host_ptr_ + offset, n);
}


} // Buffer_Namespace
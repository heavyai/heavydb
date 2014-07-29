/**
 * @file        Buffer.cpp
 * @author      Steven Stewart <steve@map-d.com>
 */ 
#include <cstring>
#include <cassert>
#include <iostream>
#include "Buffer.h"
#include "../../Shared/macros.h"

namespace Buffer_Namespace {

Buffer::Buffer(mapd_addr_t host_ptr, mapd_size_t numPages, mapd_size_t pageSize) {
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

bool Buffer::write(mapd_size_t offset, mapd_size_t n, mapd_addr_t src) {
	assert(n > 0);

	// check for buffer overflow
	if ((length_ + n) > size())
		return false;

	// write source contents to buffer
	assert(host_ptr_ && src);
	memcpy(host_ptr_ + offset, src, n);
	//length_ += n; // TODD: if not appending you aren't adding to length?

	// update dirty flags
	dirty_ = true;
	mapd_size_t firstPage = offset / pageSize_;
	mapd_size_t lastPage = (offset + n) / pageSize_;

	for (mapd_size_t i = firstPage; i < lastPage; ++i)
		pages_[i]->dirty = true;

	return true;
}

bool Buffer::append(mapd_size_t n, mapd_addr_t src) {
	assert(n > 0 && src);
	if ((length_ + n) >= size())
		return false;
	if (write(length_, n, src)) {
		length_ += n;
		return true;
	}
	return false;
}
   
bool Buffer::copy(mapd_size_t offset, mapd_size_t n, mapd_addr_t dest) {
	assert(n > 0 && dest);
	if ((n + offset) >= length_)
		return false;
	memcpy(dest, host_ptr_ + offset, n);
}

std::vector<bool> Buffer::getDirty() {
	std::vector<bool> dirtyFlags;
	for (int i = 0; i < pages_.size(); ++i)
		if (pages_[i]->dirty)
			dirtyFlags.push_back(true);
	return dirtyFlags;
}

void Buffer::print() {
	printf("host pointer = %p\n", host_ptr_);
	printf("page size    = %lu\n", pageSize_);
	printf("# of pages   = %lu\n", pages_.size());
	printf("length       = %lu\n", length_);
	printf("pin count    = %d\n", pins_);
	printf("dirty        = %s\n", dirty_ ? "true" : "false");
}


} // Buffer_Namespace

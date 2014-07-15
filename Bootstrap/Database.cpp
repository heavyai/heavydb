/**
 * @file Database.cpp
 * @author Steven Stewart
 *
 */
#include <iostream>
#include "Database.h"
#include "../DataMgr/File/FileMgr.h"
#include "../DataMgr/Buffer/BufferMgr.h"
#include "../Shared/types.h"

using namespace Database_Namespace;

Database::Database() {
	fm_ = new FileMgr(".");
	bm_ = new BufferMgr((mapd_size_t)CONFIG_HOST_MEM, fm_);
}

Database::~Database() {
	delete bm_;
	delete fm_;
}

 int main() {

 	return EXIT_SUCCESS;
 }
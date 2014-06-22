/**
 *  File:       File.cpp
 *  Author(s):  steve@map-d.com
 * 
 * 
 *  Notes: 
 * 
 *  In order to synchronize a file's in-core state with disk, the fsync function
 *  is used, which is defined in unistd.h. This is specific to POSIX operating
 *  systems, such as Unix/Linux.
 */
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include "File.h"
#include "../../Shared/errors.h"

using std::cerr;
using std::endl;
    
/**
 * Method:      File() - constructor
 * Author(s):   steve@map-d.com
 * 
 */
File::File(size_t blockSize) {
    blockSize_ = blockSize;
	f_ = NULL;
}

/**
 * Method:      ~File() - deconstructor
 * Author:      steve@map-d.com
 *
 * Ensures that the file handle is closed.
 *
 */
File::~File() {
    if (f_ != NULL)
        close();
	f_ = NULL;
}

/**
 * Method:      open()
 * Author(s):   steve@map-d.com
 * 
 * This method will open the specified file (fname). The handle to the file
 * is saved in File object's state. The client should call close() when 
 * finished with the file.
 * 
 * Note that if "create" is true, then the file will be created and the contents
 * of the file, if it already existed, are discarded. By default, this is assumed
 * false.
 * 
 * @param fname - the name of the file to be opened (or created) 
 * @param create - a flag for whether or not to create the file if it doesn't exist
 * @return - MAPD_ERR_FILE_OPEN if unable to open the file; otherwise, MAPD_SUCCESS
 */
mapd_err_t File::open(const std::string &fname, bool create) {
    if (create) {
        f_ = fopen(fname.c_str(), "w+b"); // creates a new file for updates
        fileSize_ = 0;
    }
    else {
        f_ = fopen(fname.c_str(), "r+b"); // opens existing file for updates
        fseek(f_, 0L, SEEK_END);
        fileSize_ = ftell(f_);
        fseek(f_, 0L, SEEK_SET);
    }
    if (f_ == NULL) {
        cerr << "[" << __FUNCTION__ << ":" << __LINE__ << "] ERROR: unable to open file \"" << fname.c_str() << "\"" << endl;
        return MAPD_ERR_FILE_OPEN;
    }
    return MAPD_SUCCESS;
}

/**
 * Method:      close()
 * Author(s):   steve@map-d.com
 * 
 * This method closes the file pointed to by this->f_.
 * 
 * @return MAPD_SUCCES on successful close; otherwise, MAPD_ERR_FILE_CLOSE
 */
mapd_err_t File::close() {
    return (fclose(f_) == 0) ? MAPD_SUCCESS : MAPD_ERR_FILE_CLOSE; 
}

/**
 * Method:      read()
 * Author(s):   steve@map-d.com
 * 
 * This function reads n bytes into buf beginning at position pos of the file
 * pointed to by file handle this->f_.
 * 
 * @param pos - the position (offset) into the file from where reading begins
 * @param n - the number of bytes to be read from the file
 * @param buf - the destination buffer for the read from file
 * @return MAPD_SUCCESS, MAPD_FAILURE, MAPD_ERR_FILE_READ
 */
mapd_err_t File::read(_mapd_size_t pos, _mapd_size_t n, void *buf) const {
    assert(buf);
    
    // error if attempting to read past the end of the file
    if (pos + n > fileSize_) {
        fprintf(stderr, "[%s:%d] ERROR: unable to read bytes beyond the end of the file (offset=%u bytes=%u size=%lu).\n", __func__, __LINE__, pos, n, fileSize_);
        return MAPD_ERR_FILE_READ;
    }
    
    // if the file is not open, then return an error
    if (!f_) return MAPD_FAILURE;
    
    fseek(f_, pos, SEEK_SET);
    if (fread(buf, sizeof(_byte_t), n, f_) < 1) {
        fprintf(stderr, "[%s:%d] ERROR: unable to read files.\n",  __func__, __LINE__);
        return MAPD_ERR_FILE_READ;
    }

    return MAPD_SUCCESS;
}

/**
 * Method:      write()
 * Author(s):   steve@map-d.com
 * 
 * This function writes n bytes from buf to the position pos of the file
 * pointed to by file handle this->f_.
 * 
 * @param pos - the position (offset) into the file where writing begins
 * @param n - the number of bytes to be written
 * @param buf - the source buffer that contains the data to be written to file
 * @return MAPD_SUCCESS, MAPD_FAILURE, MAPD_ERR_FILE_WRITE
 */
mapd_err_t File::write(_mapd_size_t pos, _mapd_size_t n, void *buf) {
    assert(buf);

    // if the file is not open, then return an error
    if (!f_) return MAPD_FAILURE;
    
    // write n bytes from the buffer to the file
    fseek(f_, pos, SEEK_SET);
    int bytesWritten = fwrite(buf, sizeof(_byte_t), n, f_);
    if (bytesWritten < 1) {
        cerr << "[" << __FUNCTION__ << ":" << __LINE__ << "] ERROR: unable to write to file \"" << fileName_.c_str() << "\"" << endl;
        return MAPD_ERR_FILE_WRITE;
    }
    
    // update file size
    fileSize_ += bytesWritten;
    
    return MAPD_SUCCESS;
}

mapd_err_t File::append(_mapd_size_t n, void *buf) {
    return write(fileSize_, blockSize_, buf);
}

/**
 * Method:      readBlock()
 * Author(s):   steve@map-d.com
 * 
 * This method reads a specific block from the file into the buffer.
 * 
 * @param blockNum - the block number from which data is being read into the buffer
 * @param buf - the destination buffer to which data is being written from the block
 * @return 
 */
mapd_err_t File::readBlock(_mapd_size_t blockNum, void *buf) const {
    return read(blockNum * blockSize_, blockSize_, buf);
}

/**
 * Method:      writeBlock()
 * Author(s):   steve@map-d.com
 * 
 * This method writes the buffer to a specific block in the file.
 * 
 * @param blockNum - the block number to which data is written within the file
 * @param buf - the source buffer whose contents is written to the block
 * @return 
 */
mapd_err_t File::writeBlock(_mapd_size_t blockNum, void *buf) {
    return write(blockNum * blockSize_, blockSize_, buf);
}

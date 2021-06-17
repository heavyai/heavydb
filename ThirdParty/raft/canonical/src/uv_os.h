/* Operating system related utilities. */

#ifndef UV_OS_H_
#define UV_OS_H_

#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <uv.h>

/* For backward compat with older libuv */
#if !defined(UV_FS_O_RDONLY)
#define UV_FS_O_RDONLY O_RDONLY
#endif

#if !defined(UV_FS_O_DIRECTORY)
#define UV_FS_O_DIRECTORY O_DIRECTORY
#endif

#if !defined(UV_FS_O_WRONLY)
#define UV_FS_O_WRONLY O_WRONLY
#endif

#if !defined(UV_FS_O_RDWR)
#define UV_FS_O_RDWR O_RDWR
#endif

#if !defined(UV_FS_O_CREAT)
#define UV_FS_O_CREAT O_CREAT
#endif

#if !defined(UV_FS_O_TRUNC)
#define UV_FS_O_TRUNC O_TRUNC
#endif

#if !defined(UV_FS_O_EXCL)
#define UV_FS_O_EXCL O_EXCL
#endif

#if !defined(UV_FS_O_DIRECT)
#define UV_FS_O_DIRECT O_DIRECT
#endif

#if !defined(UV_FS_O_NONBLOCK)
#define UV_FS_O_NONBLOCK O_NONBLOCK
#endif

/* Maximum size of a full file system path string. */
#define UV__PATH_SZ 1024

/* Maximum length of a filename string. */
#define UV__FILENAME_LEN 128

/* Length of path separator. */
#define UV__SEP_LEN 1 /* strlen("/") */

/* True if STR's length is at most LEN. */
#define LEN_AT_MOST_(STR, LEN) (strnlen(STR, LEN + 1) <= LEN)

/* Maximum length of a directory path string. */
#define UV__DIR_LEN (UV__PATH_SZ - UV__SEP_LEN - UV__FILENAME_LEN - 1)

/* True if the given DIR string has at most UV__DIR_LEN chars. */
#define UV__DIR_HAS_VALID_LEN(DIR) LEN_AT_MOST_(DIR, UV__DIR_LEN)

/* True if the given FILENAME string has at most UV__FILENAME_LEN chars. */
#define UV__FILENAME_HAS_VALID_LEN(FILENAME) \
    LEN_AT_MOST_(FILENAME, UV__FILENAME_LEN)

/* Portable open() */
int UvOsOpen(const char *path, int flags, int mode, uv_file *fd);

/* Portable close() */
int UvOsClose(uv_file fd);

/* TODO: figure a portable abstraction. */
int UvOsFallocate(uv_file fd, off_t offset, off_t len);

/* Portable truncate() */
int UvOsTruncate(uv_file fd, off_t offset);

/* Portable fsync() */
int UvOsFsync(uv_file fd);

/* Portable fdatasync() */
int UvOsFdatasync(uv_file fd);

/* Portable stat() */
int UvOsStat(const char *path, uv_stat_t *sb);

/* Portable write() */
int UvOsWrite(uv_file fd,
              const uv_buf_t bufs[],
              unsigned int nbufs,
              int64_t offset);

/* Portable unlink() */
int UvOsUnlink(const char *path);

/* Portable rename() */
int UvOsRename(const char *path1, const char *path2);

/* Join dir and filename into a full OS path. */
void UvOsJoin(const char *dir, const char *filename, char *path);

/* TODO: figure a portable abstraction. */
int UvOsIoSetup(unsigned nr, aio_context_t *ctxp);
int UvOsIoDestroy(aio_context_t ctx);
int UvOsIoSubmit(aio_context_t ctx, long nr, struct iocb **iocbpp);
int UvOsIoGetevents(aio_context_t ctx,
                    long min_nr,
                    long max_nr,
                    struct io_event *events,
                    struct timespec *timeout);
int UvOsEventfd(unsigned int initval, int flags);
int UvOsSetDirectIo(uv_file fd);

/* Format an error message caused by a failed system call or stdlib function. */
#define UvOsErrMsg(ERRMSG, SYSCALL, ERRNUM)              \
    {                                                    \
        ErrMsgPrintf(ERRMSG, "%s", uv_strerror(ERRNUM)); \
        ErrMsgWrapf(ERRMSG, SYSCALL);                    \
    }

#endif /* UV_OS_H_ */

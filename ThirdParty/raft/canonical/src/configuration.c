#include "configuration.h"

#include "assert.h"
#include "byte.h"

/* Current encoding format version. */
#define ENCODING_FORMAT 1

void configurationInit(struct raft_configuration *c)
{
    c->servers = NULL;
    c->n = 0;
}

void configurationClose(struct raft_configuration *c)
{
    size_t i;
    assert(c != NULL);
    assert(c->n == 0 || c->servers != NULL);
    for (i = 0; i < c->n; i++) {
        raft_free(c->servers[i].address);
    }
    if (c->servers != NULL) {
        raft_free(c->servers);
    }
}

unsigned configurationIndexOf(const struct raft_configuration *c,
                              const raft_id id)
{
    unsigned i;
    assert(c != NULL);
    for (i = 0; i < c->n; i++) {
        if (c->servers[i].id == id) {
            return i;
        }
    }
    return c->n;
}

unsigned configurationIndexOfVoter(const struct raft_configuration *c,
                                   const raft_id id)
{
    unsigned i;
    unsigned j = 0;
    assert(c != NULL);
    assert(id > 0);

    for (i = 0; i < c->n; i++) {
        if (c->servers[i].id == id) {
            if (c->servers[i].role == RAFT_VOTER) {
                return j;
            }
            return c->n;
        }
        if (c->servers[i].role == RAFT_VOTER) {
            j++;
        }
    }

    return c->n;
}

const struct raft_server *configurationGet(const struct raft_configuration *c,
                                           const raft_id id)
{
    size_t i;
    assert(c != NULL);
    assert(id > 0);

    /* Grab the index of the server with the given ID */
    i = configurationIndexOf(c, id);

    if (i == c->n) {
        /* No server with matching ID. */
        return NULL;
    }
    assert(i < c->n);

    return &c->servers[i];
}

unsigned configurationVoterCount(const struct raft_configuration *c)
{
    unsigned i;
    unsigned n = 0;
    assert(c != NULL);
    for (i = 0; i < c->n; i++) {
        if (c->servers[i].role == RAFT_VOTER) {
            n++;
        }
    }
    return n;
}

int configurationCopy(const struct raft_configuration *src,
                      struct raft_configuration *dst)
{
    size_t i;
    int rv;
    configurationInit(dst);
    for (i = 0; i < src->n; i++) {
        struct raft_server *server = &src->servers[i];
        rv = configurationAdd(dst, server->id, server->address, server->role);
        if (rv != 0) {
            return rv;
        }
    }
    return 0;
}

int configurationAdd(struct raft_configuration *c,
                     raft_id id,
                     const char *address,
                     int role)
{
    struct raft_server *servers;
    struct raft_server *server;
    size_t i;
    assert(c != NULL);
    assert(id != 0);

    if (role != RAFT_STANDBY && role != RAFT_VOTER && role != RAFT_SPARE) {
        return RAFT_BADROLE;
    }

    /* Check that neither the given id or address is already in use */
    for (i = 0; i < c->n; i++) {
        server = &c->servers[i];
        if (server->id == id) {
            return RAFT_DUPLICATEID;
        }
        if (strcmp(server->address, address) == 0) {
            return RAFT_DUPLICATEADDRESS;
        }
    }

    /* Grow the servers array.. */
    servers = raft_realloc(c->servers, (c->n + 1) * sizeof *server);
    if (servers == NULL) {
        return RAFT_NOMEM;
    }
    c->servers = servers;

    /* Fill the newly allocated slot (the last one) with the given details. */
    server = &servers[c->n];
    server->id = id;
    server->address = raft_malloc(strlen(address) + 1);
    if (server->address == NULL) {
        return RAFT_NOMEM;
    }
    strcpy(server->address, address);
    server->role = role;

    c->n++;

    return 0;
}

int configurationRemove(struct raft_configuration *c, const raft_id id)
{
    unsigned i;
    unsigned j;
    struct raft_server *servers;
    assert(c != NULL);

    i = configurationIndexOf(c, id);
    if (i == c->n) {
        return RAFT_BADID;
    }

    assert(i < c->n);

    /* If this is the last server in the configuration, reset everything. */
    if (c->n - 1 == 0) {
        raft_free(c->servers[0].address);
        raft_free(c->servers);
        c->n = 0;
        c->servers = NULL;
        return 0;
    }

    /* Create a new servers array. */
    servers = raft_calloc(c->n - 1, sizeof *servers);
    if (servers == NULL) {
        return RAFT_NOMEM;
    }

    /* Copy the first part of the servers array into a new array, excluding the
     * i'th server. */
    for (j = 0; j < i; j++) {
        servers[j] = c->servers[j];
    }

    /* Copy the second part of the servers array into a new array. */
    for (j = i + 1; j < c->n; j++) {
        servers[j - 1] = c->servers[j];
    }

    /* Release the address of the server that was deleted. */
    raft_free(c->servers[i].address);

    /* Release the old servers array */
    raft_free(c->servers);

    c->servers = servers;
    c->n--;

    return 0;
}

size_t configurationEncodedSize(const struct raft_configuration *c)
{
    size_t n = 0;
    unsigned i;

    /* We need one byte for the encoding format version */
    n++;

    /* Then 8 bytes for number of servers. */
    n += sizeof(uint64_t);

    /* Then some space for each server. */
    for (i = 0; i < c->n; i++) {
        struct raft_server *server = &c->servers[i];
        assert(server->address != NULL);
        n += sizeof(uint64_t);            /* Server ID */
        n += strlen(server->address) + 1; /* Address */
        n++;                              /* Voting flag */
    };

    return bytePad64(n);
}

void configurationEncodeToBuf(const struct raft_configuration *c, void *buf)
{
    void *cursor = buf;
    unsigned i;

    /* Encoding format version */
    bytePut8(&cursor, ENCODING_FORMAT);

    /* Number of servers. */
    bytePut64Unaligned(&cursor, c->n); /* cursor might not be 8-byte aligned */

    for (i = 0; i < c->n; i++) {
        struct raft_server *server = &c->servers[i];
        assert(server->address != NULL);
        bytePut64Unaligned(&cursor, server->id); /* might not be aligned */
        bytePutString(&cursor, server->address);
        assert(server->role < 255);
        bytePut8(&cursor, (uint8_t)server->role);
    };
}

int configurationEncode(const struct raft_configuration *c,
                        struct raft_buffer *buf)
{
    assert(c != NULL);
    assert(buf != NULL);

    /* The configuration can't be empty. */
    assert(c->n > 0);

    buf->len = configurationEncodedSize(c);
    buf->base = raft_malloc(buf->len);
    if (buf->base == NULL) {
        return RAFT_NOMEM;
    }

    configurationEncodeToBuf(c, buf->base);

    return 0;
}

int configurationDecode(const struct raft_buffer *buf,
                        struct raft_configuration *c)
{
    const void *cursor;
    size_t i;
    size_t n;

    assert(c != NULL);
    assert(buf != NULL);

    /* TODO: use 'if' instead of assert for checking buffer boundaries */
    assert(buf->len > 0);

    /* Check that the target configuration is empty. */
    assert(c->n == 0);
    assert(c->servers == NULL);

    cursor = buf->base;

    /* Check the encoding format version */
    if (byteGet8(&cursor) != ENCODING_FORMAT) {
        return RAFT_MALFORMED;
    }

    /* Read the number of servers. */
    n = (size_t)byteGet64Unaligned(&cursor);

    /* Decode the individual servers. */
    for (i = 0; i < n; i++) {
        raft_id id;
        const char *address;
        int role;
        int rv;

        /* Server ID. */
        id = byteGet64Unaligned(&cursor);

        /* Server Address. */
        address = byteGetString(
            &cursor,
            buf->len - (size_t)((uint8_t *)cursor - (uint8_t *)buf->base));
        if (address == NULL) {
            return RAFT_MALFORMED;
        }

        /* Role code. */
        role = byteGet8(&cursor);

        rv = configurationAdd(c, id, address, role);
        if (rv != 0) {
            return rv;
        }
    }

    return 0;
}

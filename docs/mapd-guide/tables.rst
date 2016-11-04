Tables
======

``CREATE TABLE``
~~~~~~~~~~~~~~~~

::

    CREATE TABLE [IF NOT EXISTS] <table>
      (<column> <type> [NOT NULL] [ENCODING <encoding spec>], ...)
      [WITH (<property> = value, ...)];

``<type>`` supported include:

-  BOOLEAN
-  SMALLINT
-  INT[EGER]
-  BIGINT
-  FLOAT \| REAL
-  DOUBLE [PRECISION]
-  [VAR]CHAR(length)
-  TEXT
-  TIME
-  TIMESTAMP
-  DATE

``<encoding spec>`` supported include:

-  DICT: Dictionary encoding on string columns (The Default for TEXT
   columns).
-  NONE: No encoding. Only valid on TEXT columns. No Dictionary will be
   created. Aggregate operations will not be possible on this column
   type
-  FIXED(bits): Fixed length encoding of integer or timestamp columns.

The ``<property>`` in the optional WITH clause can be

-  ``fragment_size``: number of rows per fragment which is a unit of the
   table for query processing. It defaults to 32 million rows and is not
   expected to be changed.
-  ``max_rows``: maximum number of rows allowed in the table. When this limit
   is reached, the oldest fragment will be removed. The default is 2^62.
-  ``page_size``: number of bytes for an I/O page. This defaults to 1MB
   and does not need to be changed.

Example:

::

    CREATE TABLE IF NOT EXISTS tweets (
      tweet_id BIGINT NOT NULL,
      tweet_time TIMESTAMP NOT NULL ENCODING FIXED(32),
      lat REAL,
      lon REAL,
      sender_id BIGINT NOT NULL,
      sender_name TEXT NOT NULL ENCODING DICT,
      location TEXT ENCODING DICT,
      source TEXT ENCODING DICT,
      reply_to_user_id BIGINT,
      reply_to_tweet_id BIGINT,
      lang TEXT ENCODING DICT,
      followers INT,
      followees INT,
      tweet_count INT,
      join_time TIMESTAMP ENCODING FIXED(32),
      tweet_text TEXT,
      state TEXT ENCODING DICT,
      county TEXT ENCODING DICT,
      place_name TEXT,
      state_abbr TEXT ENCODING DICT,
      county_state TEXT ENCODING DICT,
      origin TEXT ENCODING DICT);

``ALTER TABLE``
~~~~~~~~~~~~~~~

::

    ALTER TABLE <table> RENAME TO <table>;
    ALTER TABLE <table> RENAME COLUMN <column> TO <column>;

Example:

::

    ALTER TABLE tweets RENAME TO retweets;
    ALTER TABLE retweets RENAME COLUMN source TO device;

``DROP TABLE``
~~~~~~~~~~~~~~

::

    DROP TABLE [IF EXISTS] <table>;

Example:

::

    DROP TABLE IF EXISTS tweets;

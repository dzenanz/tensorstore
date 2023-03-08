.. _zip-kvstore-driver:

``zip`` Key-Value Store driver
=================================

The ``zip`` driver stores key-value pairs in a zip file.
The zip file can reside in memory or on the local file system.
The main need for zip store is for use in JavaScript and WebAssembly,
where a single file can be passed around as a blob,
and a directory is very inconvenient (to say the least).
It includes full support for multi-key transactions.

.. json:schema:: kvstore/zip

.. json:schema:: Context.zip_key_value_store

.. json:schema:: KvStoreUrl/zip


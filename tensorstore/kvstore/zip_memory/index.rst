.. _zip_memory-kvstore-driver:

``zip_memory`` Key-Value Store driver
=================================

The ``zip_memory`` driver stores key-value pairs in a zip file.
The zip file can reside in memory or on the local file system.
The main need for zip store is for use in JavaScript and WebAssembly,
where a single file can be passed around as a blob,
and a directory is very inconvenient (to say the least).

.. json:schema:: kvstore/zip_memory

.. json:schema:: Context.zip_memory_key_value_store

.. json:schema:: KvStoreUrl/zip_memory


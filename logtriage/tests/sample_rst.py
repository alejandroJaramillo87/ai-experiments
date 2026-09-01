"""A reStructuredText fixture reproducing the constructs the real manual uses.

Hand-written but not invented: every directive here appears in the MongoDB
documentation, and each one is something a naive splitter gets wrong. Shared
between the chunker tests and the retrieval tests so both exercise the same
document.
"""

from __future__ import annotations

MANUAL = """\
.. _log-messages:

============
Log Messages
============

.. default-domain:: mongodb

.. contents:: On this page
   :local:
   :backlinks: none

Starting in MongoDB 4.4, structured log messages are output as :term:`JSON`
objects. See :ref:`the log message reference <log-message-format>` for the
full schema, or :doc:`/reference/parameters` for the parameters involved.

Structured Message Format
-------------------------

Each message is a single ``JSON`` object with a fixed set of fields.

.. code-block:: javascript

   {
     "t": { "$date": "2026-08-30T14:22:31.123+00:00" },
     "s": "I",
     "c": "COMMAND",
     "id": 51803,
     "msg": "Slow query"
   }

.. include:: /includes/fact-log-message-fields.rst

Slow Query Logging
------------------

.. _slow-query-threshold:

Operations exceeding the threshold are logged with id 51803.

.. note::

   The threshold defaults to 100 milliseconds.

Configuring the Threshold
~~~~~~~~~~~~~~~~~~~~~~~~~

Set the threshold with ``setParameter``.

.. tabs::

   .. tab:: Command Line
      :tabid: cli

      Pass the option at startup.

      .. code-block:: bash

         mongod --slowms 50

   .. tab:: Shell
      :tabid: shell

      Set it at runtime.

      .. code-block:: javascript

         db.setProfilingLevel(1, { slowms: 50 })

Reference Table
---------------

+------------+---------------------------+
| Component  | Meaning                   |
+============+===========================+
| COMMAND    | Database commands         |
+------------+---------------------------+
| NETWORK    | Connection lifecycle      |
+------------+---------------------------+
"""

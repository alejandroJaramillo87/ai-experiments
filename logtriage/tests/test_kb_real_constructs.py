"""Regressions for defects the real manual exposed and the fixtures could not.

Every case here is a verbatim construct from the MongoDB Database Manual, kept
as a fixture so the regression runs offline in a bare checkout. They are
separated from test_kb.py because their provenance is the point: the hand-written
fixtures in that file were written by the same person as the parser and encode
the same assumptions, so they agreed with it about five things that were wrong.

What each one caught, measured over the manual at commit 88ec8192b924:

  - 22 of 40 code blocks on reference/log-messages.txt began with a
    `:copyable: false` option line presented as the first line of the command.
  - 44 role targets in 111 chunks rendered as `~bin.mongod` rather than `mongod`.
  - `{+version+}`-style constants reached chunk text unexpanded.
  - 78 include targets written without an extension resolved as missing, which
    silently dropped their content from the pages that included them.
  - Page-level field lists such as `:orphan:` became chunks whose entire content
    was the word "orphan".
"""

from __future__ import annotations

from logtriage.kb.chunk import chunk_document
from logtriage.kb.constants import expand, load_constants
from logtriage.kb.parse import count_residual_roles, parse_rst, strip_roles

# reference/log-messages.txt, verbatim.
LOG_MESSAGES = """\
.. _log-messages-ref:

============
Log Messages
============

.. default-domain:: mongodb

.. facet::
   :name: programming_language
   :values: shell

.. contents:: On this page
   :local:
   :backlinks: none

Structured Logging
------------------

:binary:`~bin.mongod` / :binary:`~bin.mongos` instances output all log messages
in :ref:`structured JSON format <log-message-json-output-format>`. To learn
more, run the :dbcommand:`getLog` command or see :doc:`/reference/log-messages`.

.. example::

   The following is an example log message in JSON format:

   .. code-block:: javascript
      :copyable: false

      {"t":{"$date":"2020-05-01T15:16:17.180+00:00"},"s":"I", "c":"NETWORK", "id":12345, "ctx":"listener", "svc": "R", "msg":"Listening on", "attr":{"address":"127.0.0.1"}}

Slow Query Logging
------------------

.. include:: /includes/fact-log-redaction.rst

MongoDB {+version+} logs slow operations above :ref:`the threshold
<profiling-threshold>`.
"""

ORPHAN_PAGE = """\
:orphan:

Some Page
=========

Body text that should survive.
"""


class TestRoleRendering:
    def test_tilde_targets_render_as_their_last_component(self) -> None:
        """Sphinx's `~` means "show only the final component". Without it the
        manual's own name for the server -- the single most likely search term
        in a log-triage corpus -- is stored as `~bin.mongod`."""
        text, references = strip_roles(
            ":binary:`~bin.mongod` and :binary:`~bin.mongos`"
        )
        assert text == "mongod and mongos"
        # The full target is kept, because it is what a cross-reference resolves
        # against even though it is not what a reader should see.
        assert references == ("bin.mongod", "bin.mongos")

    def test_a_method_target_keeps_its_call_syntax(self) -> None:
        text, _ = strip_roles(":method:`~cursor.hint()`")
        assert text == "hint()"

    def test_a_role_with_display_text_is_unaffected(self) -> None:
        text, references = strip_roles(":ref:`structured JSON format <log-fmt>`")
        assert text == "structured JSON format"
        assert references == ("log-fmt",)


class TestCodeBlocks:
    def test_option_lines_are_not_part_of_the_code(self) -> None:
        """`:copyable: false` is rendering metadata. Presented as the first line
        of a code block it reads as part of the command, which is the one thing
        a reader is most likely to copy."""
        blocks = parse_rst(LOG_MESSAGES, "reference/log-messages.txt")
        code = [b for b in blocks if b.kind == "code"]
        assert code, "the example code block was not parsed"
        for block in code:
            assert not block.text.lstrip().startswith(":copyable:")
            assert not block.text.lstrip().startswith(":")
        assert code[0].text.startswith('{"t":{"$date"')
        assert code[0].language == "javascript"


class TestConstants:
    def test_constants_expand_before_roles_are_stripped(self) -> None:
        """snooty.toml defines constants whose values are themselves roles, so
        expanding after role stripping would leave role syntax in the output of
        the pass that removes it."""
        constants = {"mongosh": ":binary:`~bin.mongosh`", "version": "8.3"}
        expanded, missing = expand("Run {+mongosh+} against {+version+}.", constants)
        assert missing == ()
        text, _ = strip_roles(expanded)
        assert text == "Run mongosh against 8.3."
        assert count_residual_roles(text) == 0

    def test_an_unknown_constant_is_left_legible_and_reported(self) -> None:
        """Blanking it would be unrecoverable: nobody reading the chunk later
        could tell a word had been removed."""
        expanded, missing = expand("MongoDB {+nosuch+} here", {})
        assert expanded == "MongoDB {+nosuch+} here"
        assert missing == ("nosuch",)

    def test_constants_load_from_a_snooty_toml(self, tmp_path) -> None:
        toml = tmp_path / "snooty.toml"
        toml.write_text(
            'name = "docs"\n\n[constants]\nversion = "8.3"\nport = 27017\n',
            encoding="utf-8",
        )
        constants = load_constants(toml)
        assert constants["version"] == "8.3"
        # A number in the toml still splices into prose as text.
        assert constants["port"] == "27017"


class TestIncludes:
    def test_an_unresolved_include_is_named_rather_than_dropped(self) -> None:
        chunks = chunk_document(LOG_MESSAGES, "reference/log-messages.txt")
        assert any(
            "includes /includes/fact-log-redaction.rst" in c.text for c in chunks
        )

    def test_a_resolved_include_is_parsed_into_the_page(self) -> None:
        """Four pages in five include something. An include left as a reference
        is a page whose explanation is missing."""

        def resolver(target, options, parent):
            assert target == "/includes/fact-log-redaction.rst"
            return "Log redaction removes the values of all fields.\n"

        chunks = chunk_document(
            LOG_MESSAGES, "reference/log-messages.txt", resolver=resolver
        )
        joined = "\n".join(c.text for c in chunks)
        assert "Log redaction removes the values of all fields." in joined
        assert "(includes " not in joined

    def test_a_cycle_stops_instead_of_recurring(self) -> None:
        def resolver(target, options, parent):
            return ".. include:: /loop.rst\n"

        chunks = chunk_document(
            ".. include:: /loop.rst\n", "page.txt", resolver=resolver
        )
        assert any("includes /loop.rst" in c.text for c in chunks)

    def test_start_after_and_end_before_bound_the_fragment(self) -> None:
        from logtriage.kb.corpus import _slice

        text = "before\nSTART\nkeep this\nEND\nafter"
        assert _slice(text, "START", "END").strip() == "keep this"
        # A bound that does not appear leaves the fragment whole rather than
        # silently emptying it.
        assert _slice(text, "MISSING", None) == text


class TestFieldLists:
    def test_a_page_level_field_list_is_not_prose(self) -> None:
        chunks = chunk_document(ORPHAN_PAGE, "about.txt")
        assert all("orphan" not in c.text for c in chunks)
        assert any("Body text that should survive." in c.text for c in chunks)


class TestWholePage:
    def test_the_real_page_chunks_cleanly(self) -> None:
        """The whole pipeline in the order Corpus runs it: expand constants,
        resolve includes, then parse. Constant expansion is deliberately not the
        parser's job -- it needs snooty.toml, which is a corpus fact -- so a test
        that skips it is testing a different pipeline than the one that runs."""
        expanded, missing = expand(LOG_MESSAGES, {"version": "8.3"})
        assert missing == ()

        chunks = chunk_document(
            expanded,
            "reference/log-messages.txt",
            resolver=lambda target, options, parent: "Redaction removes values.\n",
        )
        assert chunks
        assert all(c.heading_path for c in chunks[1:])
        assert sum(count_residual_roles(c.text) for c in chunks) == 0
        assert all("{+" not in c.text for c in chunks)
        assert not any("~bin." in c.text for c in chunks)
        joined = "\n".join(c.text for c in chunks)
        assert "MongoDB 8.3 logs slow operations" in joined
        assert "Redaction removes values." in joined

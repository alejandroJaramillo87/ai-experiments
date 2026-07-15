"""Subcommand modules for the unified interp CLI (python -m interp).

Each module exposes configure(sub) to register its subparser and a
cmd(args) -> int handler; heavy imports stay lazy inside cmd.
"""

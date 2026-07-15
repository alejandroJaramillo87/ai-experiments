"""Subcommand modules for the unified training CLI (python -m training).

Each module exposes configure(sub), which registers its subparser and binds
func to a cmd(args) -> int entry point. Modules keep their imports light;
heavy dependencies (torch, transformers) load lazily inside cmd().
"""

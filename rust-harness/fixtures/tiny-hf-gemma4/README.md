# tiny-hf-gemma4

The tiny Gemma 4 that `transformers` itself constructed and saved, with `capture.json`
holding the reference forward pass over it and `manifest.json` the sha256 of every
file, the generator, the seed and the torch and `transformers` versions. Written by
`../build_tiny_reference.py` on the workstation; the Tier T oracle for
`runtime/tests/ladder_tiny.rs` and the fixture behind every offline test that needs a
real Gemma 4 rather than a synthetic one.

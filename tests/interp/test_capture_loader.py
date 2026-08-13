"""Which nnsight wrapper a checkpoint's layout requires.

Gemma 4 registers as AutoModelForImageTextToText, and nnsight 0.7.0 refuses to
build such a checkpoint as a LanguageModel -- it names VisionLanguageModel in
the error instead. The choice has to come from the config rather than from
catching that refusal, and it has to agree with the hook path: the same
`text_config` nesting that selects the wrapper also selects the composite
decoder template in interp/families.py. A loader and a hook path that disagree
about the layout is how every Gemma 4 hook once pointed at model.layers.

nnsight and transformers are workbench dependencies absent from the sandbox,
so these tests exercise the selection, not a real load.
"""

from __future__ import annotations

import sys
import types

import pytest

from interp.probes.capture import CaptureError, _tokenizer_of


class Config:
    def __init__(self, text_config=None):
        if text_config is not None:
            self.text_config = text_config


@pytest.fixture
def fake_transformers(monkeypatch):
    """Stand in for transformers.AutoConfig, which the sandbox lacks."""

    def install(config):
        module = types.ModuleType("transformers")
        module.AutoConfig = types.SimpleNamespace(
            from_pretrained=lambda model_id, revision="main": config
        )
        monkeypatch.setitem(sys.modules, "transformers", module)

    return install


class TestCompositeDetection:
    def test_a_nested_text_config_is_composite(self, fake_transformers):
        from interp.probes.capture import is_composite_checkpoint

        fake_transformers(Config(text_config=Config()))
        assert is_composite_checkpoint("/app/models/hf/gemma-4-E4B") is True

    def test_a_flat_config_is_not(self, fake_transformers):
        from interp.probes.capture import is_composite_checkpoint

        fake_transformers(Config())
        assert is_composite_checkpoint("/app/models/hf/some-text-model") is False


class TestTokenizerResolution:
    def test_prefers_the_wrapper_s_own_tokenizer(self):
        model = types.SimpleNamespace(tokenizer="tok")
        assert _tokenizer_of(model) == "tok"

    def test_falls_back_to_the_processor_s_tokenizer(self):
        # A vision-language wrapper fronts a processor bundling an image
        # processor with the tokenizer; only the tokenizer half is wanted.
        model = types.SimpleNamespace(
            processor=types.SimpleNamespace(tokenizer="tok"), tokenizer=None
        )
        assert _tokenizer_of(model) == "tok"

    def test_neither_is_an_error_naming_the_wrapper(self):
        # Silently proceeding would lose the fingerprint the manifest records
        # and the padding mask that finds the last real position.
        class Wrapper:
            pass

        with pytest.raises(CaptureError, match="Wrapper"):
            _tokenizer_of(Wrapper())

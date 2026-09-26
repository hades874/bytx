from __future__ import annotations

from pathlib import Path

import pytest

from bytx.config import load_config

SHIPPED_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"

_TWO_MODELS = """
[asr.models.bn-model]
source = "org/bn"
language = "bn"
[asr.models.multi]
source = "org/multi"
language = ""
max_clip_seconds = 15
"""


def write_config(tmp_path: Path, body: str):
    (tmp_path / "config.toml").write_text(body, encoding="utf-8")
    return load_config(tmp_path / "config.toml")


# --- the shipped config.toml -----------------------------------------------------

def test_shipped_config_routes_both_languages():
    cfg = load_config(SHIPPED_CONFIG)
    assert cfg.asr.model_for_language("bn").key == "tugstugi-medium-bn"
    assert cfg.asr.model_for_language("en").key == "large-v3-turbo"


def test_shipped_config_models_are_pinned_or_multilingual_as_intended():
    cfg = load_config(SHIPPED_CONFIG)
    assert cfg.asr.models["tugstugi-medium-bn"].language == "bn"
    assert cfg.asr.models["large-v3-turbo"].multilingual
    # Bangla keeps the conservative global clip length; English gets more context per clip.
    assert cfg.asr.models["tugstugi-medium-bn"].max_clip_seconds is None
    assert cfg.asr.models["large-v3-turbo"].max_clip_seconds == 15


def test_shipped_config_summaries_follow_the_video_language():
    assert load_config(SHIPPED_CONFIG).summarize.language == "auto"


def test_shipped_config_requires_both_models_for_setup_and_doctor():
    cfg = load_config(SHIPPED_CONFIG)
    assert cfg.asr.required_model_keys() == ["tugstugi-medium-bn", "large-v3-turbo"]


# --- backward compatibility: a config from before English support ------------------

_LEGACY = """
[asr]
model = "old-bn"
[asr.models.old-bn]
source = "org/bn"
language = "bn"
"""


def test_legacy_config_without_language_models_still_loads_and_serves_bangla(tmp_path):
    cfg = write_config(tmp_path, _LEGACY)
    assert cfg.asr.language_models == {}
    assert cfg.asr.model_for_language("bn").key == "old-bn"
    assert cfg.asr.required_model_keys() == ["old-bn"]


def test_legacy_config_gives_a_clear_error_for_english(tmp_path):
    cfg = write_config(tmp_path, _LEGACY)
    with pytest.raises(ValueError, match="No ASR model is configured for language 'en'"):
        cfg.asr.model_for_language("en")


def test_model_entry_without_language_key_defaults_to_pinned_bangla(tmp_path):
    cfg = write_config(tmp_path, '[asr.models.m]\nsource = "org/m"\n')
    assert cfg.asr.models["m"].language == "bn"
    assert not cfg.asr.models["m"].multilingual


def test_legacy_config_summaries_default_to_auto(tmp_path):
    assert write_config(tmp_path, _LEGACY).summarize.language == "auto"


# --- multilingual default model -----------------------------------------------------

def test_a_multilingual_default_model_serves_any_language_without_a_map(tmp_path):
    cfg = write_config(tmp_path, '[asr]\nmodel = "multi"\n' + _TWO_MODELS)
    assert cfg.asr.model_for_language("en").key == "multi"
    assert cfg.asr.model_for_language("bn").key == "multi"


# --- [asr.language_models] validation ------------------------------------------------

def test_language_models_map_is_parsed(tmp_path):
    body = '[asr]\nmodel = "bn-model"\n[asr.language_models]\nbn = "bn-model"\nen = "multi"\n' + _TWO_MODELS
    cfg = write_config(tmp_path, body)
    assert cfg.asr.language_models == {"bn": "bn-model", "en": "multi"}
    assert cfg.asr.model_for_language("en").key == "multi"


def test_language_models_rejects_unsupported_language(tmp_path):
    body = '[asr.language_models]\nfr = "multi"\n' + _TWO_MODELS
    with pytest.raises(ValueError, match="unsupported language 'fr'"):
        write_config(tmp_path, body)


def test_language_models_rejects_undefined_model(tmp_path):
    body = '[asr.language_models]\nen = "nope"\n' + _TWO_MODELS
    with pytest.raises(ValueError, match="'nope' is not defined"):
        write_config(tmp_path, body)


def test_language_models_rejects_model_pinned_to_another_language(tmp_path):
    body = '[asr.language_models]\nen = "bn-model"\n' + _TWO_MODELS
    with pytest.raises(ValueError, match="pinned to 'bn'"):
        write_config(tmp_path, body)


def test_unknown_default_model_is_reported(tmp_path):
    cfg = write_config(tmp_path, '[asr]\nmodel = "ghost"\n' + _TWO_MODELS)
    with pytest.raises(ValueError, match="'ghost' is not defined"):
        cfg.asr.active_model()


def test_required_model_keys_are_deduplicated_and_ordered(tmp_path):
    body = (
        '[asr]\nmodel = "bn-model"\n'
        '[asr.language_models]\nbn = "bn-model"\nen = "multi"\n' + _TWO_MODELS
    )
    assert write_config(tmp_path, body).asr.required_model_keys() == ["bn-model", "multi"]


def test_per_model_clip_seconds_are_parsed(tmp_path):
    cfg = write_config(tmp_path, _TWO_MODELS)
    assert cfg.asr.models["multi"].max_clip_seconds == 15
    assert cfg.asr.models["bn-model"].max_clip_seconds is None


# --- asr.default_language ------------------------------------------------------------

def test_default_language_is_bangla_when_unset(tmp_path):
    assert write_config(tmp_path, _LEGACY).asr.default_language == "bn"


def test_shipped_config_defaults_to_bangla_so_nothing_is_guessed():
    assert load_config(SHIPPED_CONFIG).asr.default_language == "bn"


def test_default_language_can_be_english(tmp_path):
    cfg = write_config(tmp_path, '[asr]\ndefault_language = "en"\n' + _TWO_MODELS)
    assert cfg.asr.default_language == "en"


def test_default_language_must_be_supported(tmp_path):
    with pytest.raises(ValueError, match="default_language = 'fr' is not supported"):
        write_config(tmp_path, '[asr]\ndefault_language = "fr"\n' + _TWO_MODELS)


# --- asr.auto_english_below ------------------------------------------------------------

def test_auto_english_threshold_defaults_to_the_measured_gap(tmp_path):
    # Bangla videos measured 12.9%+, English up to 8.5%: 10% sits between them.
    assert write_config(tmp_path, _LEGACY).asr.auto_english_below == 0.10
    assert load_config(SHIPPED_CONFIG).asr.auto_english_below == 0.10


def test_auto_english_threshold_is_configurable(tmp_path):
    cfg = write_config(tmp_path, '[asr]\nauto_english_below = 0.15\n' + _TWO_MODELS)
    assert cfg.asr.auto_english_below == 0.15


# --- glossary and quality settings ---------------------------------------------------

def test_shipped_config_enables_the_glossary_for_english_only():
    cfg = load_config(SHIPPED_CONFIG)
    assert cfg.asr.glossary == SHIPPED_CONFIG.parent / "glossary.txt"
    assert cfg.asr.glossary.is_file()
    assert cfg.asr.glossary_prompt_languages == ("en",)  # Bangla priming is opt-in
    assert cfg.asr.glossary_prompt_max_chars == 300


def test_glossary_defaults_when_unset(tmp_path):
    cfg = write_config(tmp_path, _TWO_MODELS)
    assert cfg.asr.glossary == tmp_path / "glossary.txt"
    assert cfg.asr.glossary_prompt_languages == ("en",)


def test_glossary_can_be_disabled_with_an_empty_path(tmp_path):
    assert write_config(tmp_path, '[asr]\nglossary = ""\n' + _TWO_MODELS).asr.glossary is None


def test_glossary_prompt_languages_are_parsed_and_validated(tmp_path):
    cfg = write_config(tmp_path, '[asr]\nglossary_prompt_languages = ["bn", "en"]\n' + _TWO_MODELS)
    assert cfg.asr.glossary_prompt_languages == ("bn", "en")
    with pytest.raises(ValueError, match="glossary_prompt_languages"):
        write_config(tmp_path, '[asr]\nglossary_prompt_languages = ["fr"]\n' + _TWO_MODELS)


def test_quality_thresholds_default_and_are_configurable(tmp_path):
    cfg = write_config(tmp_path, _TWO_MODELS)
    assert (cfg.quality.min_wpm, cfg.quality.max_wpm, cfg.quality.max_gap_seconds) == (70, 220, 10)
    custom = write_config(tmp_path, "[quality]\nmin_wpm = 90\nmax_gap_seconds = 5\n" + _TWO_MODELS)
    assert (custom.quality.min_wpm, custom.quality.max_gap_seconds, custom.quality.max_wpm) == (90, 5, 220)


def test_shipped_config_quality_matches_the_documented_defaults():
    from bytx.config import QualityConfig

    assert load_config(SHIPPED_CONFIG).quality == QualityConfig()

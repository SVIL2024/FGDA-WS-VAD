"""Guard the public path-resolution API and every published entrypoint.

Three published modules once shipped from an unfinished refactor that called
``freq_text_options.sht_root()`` / ``sht_feat_root()`` / ``official_checkpoints()``
before those helpers existed. Because the calls sat inside function bodies the
modules still *imported* cleanly, so the breakage only surfaced at evaluation
time as an ``AttributeError`` -- silently making every ShanghaiTech number and
every ``--checkpoint official`` row unreproducible from the public repo. These
tests assert the helpers exist and are callable, and that every entrypoint
imports, so that class of drift fails loudly instead.
"""
import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import freq_text_options as fto

ENTRYPOINTS = (
    'cross_eval', 'xd_train', 'xd_test', 'xd_option',
    'ucf_train', 'ucf_test', 'ucf_option',
    'union_train', 'build_union_list',
    'sht_extract', 'sht_build_gt',
    'freq_text_aug', 'freq_text_dataset', 'freq_text_events',
    'freq_text_options', 'freq_text_prompts', 'freq_text_test',
    'freq_text_text', 'freq_text_trainer',
    'model', 'probe_events', 'stage0_extract_smoke',
    'analysis_crossdomain_transfer', 'analysis_official_metric',
    'analysis_spectral_bands', 'analysis_zsad_floor',
    'analysis_gain_vs_difficulty', 'analysis_magnitude_orthogonal',
    'analysis_residual_direction', 'analysis_selective_routing',
    'analysis_selective_trust', 'analysis_spread_confound',
    'precondition_test',
)

PATH_HELPERS = ('env_path', 'data_root', 'sht_root', 'sht_feat_root',
                'official_checkpoints')


@pytest.mark.parametrize('name', PATH_HELPERS)
def test_path_helper_exists(name):
    assert hasattr(fto, name), f'freq_text_options.{name} is missing'


def test_path_helpers_are_callable():
    assert fto.data_root()
    assert fto.sht_root()
    assert fto.sht_feat_root()
    for source in ('xd', 'ucf'):
        candidates = fto.official_checkpoints(source)
        assert candidates
        assert all(isinstance(c, str) for c in candidates)


@pytest.mark.parametrize('name', ENTRYPOINTS)
def test_entrypoint_imports(name):
    importlib.import_module(name)


def test_env_precedence(monkeypatch):
    monkeypatch.setenv('RSI_DATA_ROOT', 'D:/rsi')
    monkeypatch.setenv('FGDA_DATA_ROOT', 'D:/fgda')
    assert fto.data_root() == 'D:/rsi'
    monkeypatch.delenv('RSI_DATA_ROOT')
    assert fto.data_root() == 'D:/fgda'
    monkeypatch.setenv('RSI_DATA_ROOT', '')
    assert fto.data_root() == 'D:/fgda'


def test_sht_feat_root_follows_data_root(monkeypatch):
    monkeypatch.delenv('RSI_SHT_FEAT_ROOT', raising=False)
    monkeypatch.delenv('FGDA_SHT_FEAT_ROOT', raising=False)
    monkeypatch.setenv('RSI_DATA_ROOT', 'D:/features')
    assert fto.sht_feat_root() == os.path.join('D:/features', 'SHTClipFeatures')


def test_cross_eval_shares_sht_feat_root():
    cross_eval = importlib.import_module('cross_eval')
    assert cross_eval.SHT_FEAT_ROOT == fto.sht_feat_root()

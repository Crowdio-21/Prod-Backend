"""A model exported by the SDK (crowdio-export) loads into the Foreman and can be planned over."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
exporter = pytest.importorskip("crowdio.export.exporter")

from foreman.planning.contract import Manifest, ShardRole  # noqa: E402
from foreman.planning.registry import ShardRegistry  # noqa: E402


def tiny_bert(layers=4, seed=0):
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(seed)
    config = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=layers, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=64, num_labels=2)
    return BertForSequenceClassification(config).eval()


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sdk_export")
    result = exporter.export_model(tiny_bert(), "tiny-bert", out, cut_grid=1, max_compositions=8,
                                   log=lambda *_: None)
    return result.directory if hasattr(result, "directory") else out / "tiny-bert"


def test_the_foreman_loads_an_sdk_manifest_and_numbers_the_atoms_contiguously(export_dir):
    registry = ShardRegistry.load(export_dir)
    manifest = registry.manifest
    assert isinstance(manifest, Manifest) and manifest.model_id == "tiny-bert"
    assert [a.name for a in manifest.atoms][0] == "embed" and manifest.atoms[-1].name == "head"
    assert len(manifest.atoms) == 4 + 2  # embed + 4 layers + head
    by_role = {role: [s for s in manifest.shards if s.role is role] for role in ShardRole}
    assert len(by_role[ShardRole.EMBED]) == len(by_role[ShardRole.HEAD]) == len(by_role[ShardRole.FULL]) == 1
    assert len(by_role[ShardRole.ENCODER]) == 4 * 5 // 2  # every contiguous range of 4 atoms
    embed, head, full = by_role[ShardRole.EMBED][0], by_role[ShardRole.HEAD][0], by_role[ShardRole.FULL][0]
    assert (embed.start_atom, embed.end_atom) == (0, 0) and embed.pinned
    assert (head.start_atom, head.end_atom) == (5, 5) and head.pinned
    assert (full.start_atom, full.end_atom) == (0, 5)
    assert all(1 <= s.start_atom <= s.end_atom <= 4 for s in by_role[ShardRole.ENCODER])


def test_outputs_say_which_input_of_the_next_shard_they_feed(export_dir):
    manifest = ShardRegistry.load(export_dir).manifest
    encoder = next(s for s in manifest.shards if s.role is ShardRole.ENCODER)
    bound = {o.name: o.binds_to for o in encoder.tensor_contract.outputs}
    assert bound == {"hidden_states_out": "hidden_states", "attention_mask_out": "attention_mask"}
    assert {i.name for i in encoder.tensor_contract.inputs} == {"hidden_states", "attention_mask"}
    served = json.loads(manifest.model_dump_json())  # what phones download from /api/shards/manifest
    first = next(s for s in served["shards"] if s["role"] == "encoder")
    assert first["tensor_contract"]["outputs"][0]["binds_to"] in ("hidden_states", "attention_mask")


def test_the_default_composition_tiles_all_atoms_and_files_verify(export_dir):
    registry = ShardRegistry.load(export_dir)
    ids = registry.manifest.default_composition
    shards = [registry.spec(i) for i in ids]
    assert shards[0].role is ShardRole.EMBED and shards[-1].role is ShardRole.HEAD
    assert sum(s.end_atom - s.start_atom + 1 for s in shards) == len(registry.manifest.atoms)
    assert all(registry.verify(s.shard_id) for s in registry.manifest.shards[:6])
    assert registry.path(shards[1].shard_id).is_file()


def test_the_catalog_prices_activations_for_the_planner(export_dir):
    catalog = ShardRegistry.load(export_dir).catalog
    embed = catalog.shards[next(i for i, s in catalog.shards.items() if s.role == "embed")]
    assert embed.out_act_bytes_per_token > 0 and embed.pinned
    assert catalog.full_shard_id is not None
    assert catalog.n_atoms == 6

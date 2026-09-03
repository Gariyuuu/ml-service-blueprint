"""Registry: versioning, immutability, stage pointers, promotion, rollback."""

from __future__ import annotations

import pytest

from mlservice.registry.base import ModelNotFoundError, RegistryError, VersionConflictError
from mlservice.registry.local import LocalFilesystemRegistry


@pytest.fixture
def registry_with_one(empty_registry, trained_artifact):
    entry = empty_registry.register(trained_artifact)
    return empty_registry, entry


def test_first_registration_is_v1(registry_with_one):
    _, entry = registry_with_one
    assert entry.version == "v1"


def test_versions_increment(empty_registry, trained_artifact):
    versions = [empty_registry.register(trained_artifact).version for _ in range(3)]
    assert versions == ["v1", "v2", "v3"]


def test_registering_does_not_mutate_earlier_versions(empty_registry, trained_artifact):
    first = empty_registry.register(trained_artifact)
    empty_registry.register(trained_artifact)
    assert empty_registry.load(first.model_name, "v1").metadata.version == "v1"


def test_explicit_version_can_be_requested(empty_registry, trained_artifact):
    assert empty_registry.register(trained_artifact, version="v42").version == "v42"


def test_reusing_a_version_is_refused(empty_registry, trained_artifact):
    empty_registry.register(trained_artifact, version="v5")
    with pytest.raises(VersionConflictError):
        empty_registry.register(trained_artifact, version="v5")


def test_malformed_version_strings_are_refused(empty_registry, trained_artifact):
    with pytest.raises(RegistryError, match="Invalid version"):
        empty_registry.register(trained_artifact, version="latest")


def test_a_failed_gate_blocks_registration(empty_registry, trained_artifact):
    trained_artifact.metadata = trained_artifact.metadata.model_copy(
        update={"gate_failures": ["roc_auc too low"]}
    )
    name = trained_artifact.metadata.model_name

    with pytest.raises(RegistryError, match="gates failed"):
        empty_registry.register(trained_artifact)
    assert empty_registry.list_versions(name) == [], "a refused registration must leave no trace"

    entry = empty_registry.register(trained_artifact, allow_failed_gates=True)
    assert entry.passed_gates is False


def test_list_models_and_versions(registry_with_one):
    registry, entry = registry_with_one
    assert registry.list_models() == [entry.model_name]
    assert [v.version for v in registry.list_versions(entry.model_name)] == ["v1"]


def test_listing_an_unknown_model_is_empty_not_an_error(empty_registry):
    assert empty_registry.list_versions("nope") == []


def test_getting_an_unknown_version_raises(empty_registry):
    with pytest.raises(ModelNotFoundError):
        empty_registry.get_version("nope", "v1")


def test_load_returns_a_working_artifact(registry_with_one, synthetic_frame):
    registry, entry = registry_with_one
    artifact = registry.load(entry.model_name, entry.version)
    result = artifact.predict(synthetic_frame.drop(columns=["target"]).head(5))
    assert len(result.scores) == 5


def test_promotion_moves_the_stage_pointer(registry_with_one):
    registry, entry = registry_with_one
    registry.promote(entry.model_name, "v1", "production")
    assert registry.resolve_stage(entry.model_name, "production").version == "v1"


def test_promoting_an_unknown_version_raises(registry_with_one):
    registry, entry = registry_with_one
    with pytest.raises(ModelNotFoundError):
        registry.promote(entry.model_name, "v99", "production")


def test_stages_are_independent(empty_registry, trained_artifact):
    name = trained_artifact.metadata.model_name
    empty_registry.register(trained_artifact)
    empty_registry.register(trained_artifact)
    empty_registry.promote(name, "v1", "production")
    empty_registry.promote(name, "v2", "staging")
    assert empty_registry.stages(name) == {"production": "v1", "staging": "v2"}


def test_unresolved_stage_raises_with_a_useful_message(registry_with_one):
    registry, entry = registry_with_one
    with pytest.raises(ModelNotFoundError, match="No version is assigned"):
        registry.resolve_stage(entry.model_name, "production")


def test_rollback_returns_the_stage_to_the_previous_version(empty_registry, trained_artifact):
    name = trained_artifact.metadata.model_name
    empty_registry.register(trained_artifact)
    empty_registry.register(trained_artifact)
    empty_registry.promote(name, "v1", "production")
    empty_registry.promote(name, "v2", "production")

    transition = empty_registry.rollback(name, "production")
    assert transition.to_version == "v1"
    assert transition.is_rollback
    assert empty_registry.resolve_stage(name, "production").version == "v1"


def test_rollback_without_history_is_refused(registry_with_one):
    registry, entry = registry_with_one
    with pytest.raises(ModelNotFoundError, match="nothing to roll back"):
        registry.rollback(entry.model_name, "production")


def test_rollback_from_a_first_promotion_is_refused(registry_with_one):
    """v1 was promoted from nothing; there is no earlier version to return to."""
    registry, entry = registry_with_one
    registry.promote(entry.model_name, "v1", "production")
    with pytest.raises(ModelNotFoundError, match="no earlier version"):
        registry.rollback(entry.model_name, "production")


def test_transitions_are_an_append_only_audit_log(empty_registry, trained_artifact):
    name = trained_artifact.metadata.model_name
    empty_registry.register(trained_artifact)
    empty_registry.register(trained_artifact)
    empty_registry.promote(name, "v1", "production", reason="first")
    empty_registry.promote(name, "v2", "production", reason="second")
    empty_registry.rollback(name, "production", reason="incident")

    history = empty_registry.transitions(name, "production")
    assert [t.to_version for t in history] == ["v1", "v2", "v1"]
    assert history[-1].reason == "incident"


def test_transitions_can_be_filtered_by_stage(empty_registry, trained_artifact):
    name = trained_artifact.metadata.model_name
    empty_registry.register(trained_artifact)
    empty_registry.promote(name, "v1", "production")
    empty_registry.promote(name, "v1", "staging")
    assert len(empty_registry.transitions(name, "staging")) == 1
    assert len(empty_registry.transitions(name)) == 2


def test_deleting_a_staged_version_is_refused(registry_with_one):
    registry, entry = registry_with_one
    registry.promote(entry.model_name, "v1", "production")
    with pytest.raises(RegistryError, match="still referenced"):
        registry.delete_version(entry.model_name, "v1")


def test_deleting_an_unreferenced_version_works(registry_with_one):
    registry, entry = registry_with_one
    registry.delete_version(entry.model_name, "v1")
    assert registry.list_versions(entry.model_name) == []


def test_registry_entries_report_the_stages_pointing_at_them(registry_with_one):
    registry, entry = registry_with_one
    registry.promote(entry.model_name, "v1", "production")
    assert registry.get_version(entry.model_name, "v1").stages == ["production"]


def test_path_traversal_in_a_model_name_is_refused(tmp_path):
    registry = LocalFilesystemRegistry(tmp_path)
    with pytest.raises(RegistryError, match="Invalid model name"):
        registry.list_versions("../../etc")


def test_a_registry_is_just_a_directory(registry_with_one, tmp_path):
    """The whole point of the local backend: copy the directory, keep the models."""
    import shutil

    registry, entry = registry_with_one
    copy_root = tmp_path / "copied"
    shutil.copytree(registry.root, copy_root)
    assert LocalFilesystemRegistry(copy_root).load(entry.model_name, "v1") is not None

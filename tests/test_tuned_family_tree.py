import pytest

from wdn.models.tuned_family_tree import TunedFamilyTreeConfig


def test_tuned_family_tree_config_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        TunedFamilyTreeConfig(max_leaf_nodes=1).validate()
    TunedFamilyTreeConfig().validate()

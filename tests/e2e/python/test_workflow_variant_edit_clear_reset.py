from sdk.python.goodfire.features.features import Feature


def test_workflow_variant_edit_clear_reset_8b(goodfire_client, variant_small):
    feature_group_small = goodfire_client.features.search(
        "pirate",
        model=variant_small,
    )

    selected_feature_1 = feature_group_small[0]
    selected_feature_2 = feature_group_small[1]

    assert isinstance(selected_feature_1, Feature)
    assert isinstance(selected_feature_2, Feature)

    assert len(variant_small.edits) == 0

    variant_small.set(selected_feature_1, 0.9)
    variant_small.set(selected_feature_2, 0.1)
    variant_small.set(feature_group_small[2, 5], 0.7)

    assert len(variant_small.edits) == 4

    variant_small.clear(selected_feature_1)

    assert len(variant_small.edits) == 3

    variant_small.clear([feature_group_small[2], feature_group_small[5]])

    assert len(variant_small.edits) == 1

    variant_small.reset()
    assert len(variant_small.edits) == 0


def test_workflow_variant_edit_clear_reset_70b(goodfire_client, variant_medium):
    feature_group_medium = goodfire_client.features.search(
        "pirate",
        model=variant_medium,
    )

    selected_feature_1 = feature_group_medium[0]
    selected_feature_2 = feature_group_medium[1]

    assert isinstance(selected_feature_1, Feature)
    assert isinstance(selected_feature_2, Feature)

    assert len(variant_medium.edits) == 0

    variant_medium.set(selected_feature_1, 0.9)
    variant_medium.set(selected_feature_2, 0.1)
    variant_medium.set(feature_group_medium[2, 5], 0.7)

    assert len(variant_medium.edits) == 4

    variant_medium.clear(selected_feature_1)

    assert len(variant_medium.edits) == 3

    variant_medium.clear([feature_group_medium[2], feature_group_medium[5]])

    assert len(variant_medium.edits) == 1

    variant_medium.reset()
    assert len(variant_medium.edits) == 0

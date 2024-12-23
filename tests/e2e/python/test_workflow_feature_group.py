from sdk.python.goodfire.features.features import FeatureGroup


def test_feature_group_workflow_pop_add(goodfire_client, variant_small):
    music_feature_group = goodfire_client.features.search(
        "music",
        model=variant_small,
        top_k=5,
    )

    venues_feature_group = goodfire_client.features.search(
        "venues",
        model=variant_small,
        top_k=5,
    )

    assert len(music_feature_group) == 5

    music_feature_group.add(venues_feature_group[0])

    assert len(music_feature_group) == 6

    music_feature_group.pop(0)

    assert len(music_feature_group) == 5

    music_feature_group.add(venues_feature_group[1])

    assert len(music_feature_group) == 6


def test_workflow_feature_group_pick(goodfire_client, variant_small):
    music_feature_group = goodfire_client.features.search(
        "music",
        model=variant_small,
        top_k=5,
    )

    feature_a_from_original_group = music_feature_group[0]
    feature_b_from_original_group = music_feature_group[4]

    new_feature_group = music_feature_group.pick([0, 4])

    feature_a_from_new_group = new_feature_group[0]
    feature_b_from_new_group = new_feature_group[1]

    assert feature_a_from_new_group.uuid == feature_a_from_original_group.uuid
    assert feature_b_from_new_group.uuid == feature_b_from_original_group.uuid
    assert isinstance(new_feature_group, FeatureGroup)


def test_workflow_feature_group_json_out_json_in(goodfire_client, variant_small):
    music_feature_group = goodfire_client.features.search(
        "music",
        model=variant_small,
        top_k=5,
    )

    json_out = music_feature_group.json()

    from_json_output = FeatureGroup.from_json(json_out)

    assert len(from_json_output) == len(music_feature_group)
    assert from_json_output[0].uuid == music_feature_group[0].uuid
    assert from_json_output[1].uuid == music_feature_group[1].uuid
    assert from_json_output[2].uuid == music_feature_group[2].uuid
    assert from_json_output[3].uuid == music_feature_group[3].uuid
    assert from_json_output[4].uuid == music_feature_group[4].uuid


def test_workflow_feature_group_union(goodfire_client, variant_small):
    music_feature_group = goodfire_client.features.search(
        "music",
        model=variant_small,
        top_k=5,
    )

    venues_feature_group = goodfire_client.features.search(
        "venues",
        model=variant_small,
        top_k=5,
    )

    union_music_and_venues_feature_group = music_feature_group.union(
        venues_feature_group
    )

    assert len(union_music_and_venues_feature_group) == 10
    assert isinstance(union_music_and_venues_feature_group, FeatureGroup)

    picked_group_a = union_music_and_venues_feature_group.pick([0, 1])
    picked_group_b = union_music_and_venues_feature_group.pick([2, 3])

    union_picked_group_a_and_b = picked_group_a.union(picked_group_b)

    assert len(picked_group_a) == 2
    assert len(picked_group_b) == 2
    assert len(union_picked_group_a_and_b) == 4
    assert isinstance(union_picked_group_a_and_b, FeatureGroup)
    assert (
        union_picked_group_a_and_b[0].uuid
        == union_music_and_venues_feature_group[0].uuid
    )
    assert (
        union_picked_group_a_and_b[1].uuid
        == union_music_and_venues_feature_group[1].uuid
    )
    assert (
        union_picked_group_a_and_b[2].uuid
        == union_music_and_venues_feature_group[2].uuid
    )
    assert (
        union_picked_group_a_and_b[3].uuid
        == union_music_and_venues_feature_group[3].uuid
    )

    popped_feature = union_picked_group_a_and_b.pop(0)

    assert popped_feature.uuid == union_music_and_venues_feature_group[0].uuid
    assert len(union_picked_group_a_and_b) == 3

    union_picked_group_a_and_b.add(popped_feature)

    assert len(union_picked_group_a_and_b) == 4

    json_out = union_picked_group_a_and_b.json()

    new_feature_group = FeatureGroup.from_json(json_out)

    assert len(new_feature_group) == 4

    new_feature_group_uuids = [feature.uuid for feature in new_feature_group]
    union_picked_group_a_and_b_uuids = [
        feature.uuid for feature in union_picked_group_a_and_b
    ]

    for uuid in new_feature_group_uuids:
        assert uuid in union_picked_group_a_and_b_uuids


def test_workflow_feature_group_intersection(goodfire_client, variant_small):
    music_performance_feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_small,
        top_k=10,
    )

    performing_arts_venues_feature_group = goodfire_client.features.search(
        "performing arts venues",
        model=variant_small,
        top_k=10,
    )

    uuids_from_music_performance = {
        feature.uuid for feature in music_performance_feature_group
    }
    uuids_from_performing_arts_venues = {
        feature.uuid for feature in performing_arts_venues_feature_group
    }
    uuids_from_feature_groups = (
        uuids_from_music_performance | uuids_from_performing_arts_venues
    )

    intersection_feature_group = music_performance_feature_group.intersection(
        performing_arts_venues_feature_group
    )

    uuids_from_intersection = {feature.uuid for feature in intersection_feature_group}

    for uuid in uuids_from_intersection:
        assert uuid in uuids_from_feature_groups

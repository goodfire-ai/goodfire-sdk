from numpy import ndarray as NDArray

from sdk.python.goodfire.api.features.client import FeatureActivations
from sdk.python.goodfire.features.features import Feature


def test_workflow_features_insepct_context_object_8b(goodfire_client, variant_small):
    context = goodfire_client.features.inspect(
        [
            {
                "role": "user",
                "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
            },
            {
                "role": "assistant",
                "content": "An example of a synthesizer company is Moog.",
            },
        ],
        model=variant_small,
    )

    context_matrix = context.matrix()
    assert isinstance(context_matrix, NDArray)

    context_lookup = context.lookup()
    assert isinstance(context_lookup, dict)
    assert all(isinstance(feature, Feature) for feature in context_lookup.values())

    context_token_activations = context.tokens[-3].inspect()
    assert isinstance(context_token_activations, FeatureActivations)
    assert isinstance(context_token_activations.vector(), NDArray)

    context_token_activations_lookup = context_token_activations.lookup()
    assert isinstance(context_token_activations_lookup, dict)
    assert all(
        isinstance(feature, Feature)
        for feature in context_token_activations_lookup.values()
    )

    top_3_features_activations = context.top(k=3)
    top_1_from_top_3_activations = top_3_features_activations[0]

    assert len(top_3_features_activations) == 3

    assert isinstance(top_3_features_activations, FeatureActivations)
    assert isinstance(top_3_features_activations.vector(), NDArray)

    top_5_features_activations = context.top(k=5)
    top_1_from_top_5_activations = top_5_features_activations[0]

    assert len(top_5_features_activations) == 5

    assert (
        top_1_from_top_3_activations.feature.uuid
        == top_1_from_top_5_activations.feature.uuid
    )
    assert (
        top_1_from_top_3_activations.activation
        == top_1_from_top_5_activations.activation
    )

    top_3_features_activations_lookup = top_3_features_activations.lookup()

    assert isinstance(top_3_features_activations_lookup, dict)
    assert len(top_3_features_activations_lookup) == 3
    assert all(
        isinstance(feature, Feature)
        for feature in top_3_features_activations_lookup.values()
    )

    top_3_features_activations_lookup_keys = [
        int(key) for key in top_3_features_activations_lookup.keys()
    ]

    features_lookup_result = goodfire_client.features.lookup(
        indices=top_3_features_activations_lookup_keys, model=variant_small
    )

    assert isinstance(features_lookup_result, dict)
    assert len(features_lookup_result) == 3
    assert all(
        isinstance(feature, Feature) for feature in features_lookup_result.values()
    )

    assert (
        features_lookup_result[top_3_features_activations_lookup_keys[0]].uuid
        == top_3_features_activations_lookup[
            top_3_features_activations_lookup_keys[0]
        ].uuid
    )
    assert (
        features_lookup_result[top_3_features_activations_lookup_keys[1]].uuid
        == top_3_features_activations_lookup[
            top_3_features_activations_lookup_keys[1]
        ].uuid
    )
    assert (
        features_lookup_result[top_3_features_activations_lookup_keys[2]].uuid
        == top_3_features_activations_lookup[
            top_3_features_activations_lookup_keys[2]
        ].uuid
    )


def test_workflow_features_insepct_context_object_with_specfic_features_8b(
    goodfire_client, variant_small
):
    context = goodfire_client.features.inspect(
        [
            {
                "role": "user",
                "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
            },
            {
                "role": "assistant",
                "content": "An example of a synthesizer company is Moog.",
            },
        ],
        model=variant_small,
    )

    top_1_activation_before_inspecting_specific_features = context.top(k=1)

    company_features = goodfire_client.features.search(
        "company", model=variant_small, top_k=5
    )

    context_with_specific_features = goodfire_client.features.inspect(
        [
            {
                "role": "user",
                "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
            },
            {
                "role": "assistant",
                "content": "An example of a synthesizer company is Moog.",
            },
        ],
        model=variant_small,
        features=company_features,
    )

    top_1_activation_after_inspecting_specific_features = (
        context_with_specific_features.top(k=1)
    )

    assert (
        top_1_activation_after_inspecting_specific_features[0].feature.uuid
        != top_1_activation_before_inspecting_specific_features[0].feature.uuid
    )


def test_workflow_features_insepct_context_object_70b(goodfire_client, variant_medium):
    context = goodfire_client.features.inspect(
        [
            {
                "role": "user",
                "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
            },
            {
                "role": "assistant",
                "content": "An example of a synthesizer company is Moog.",
            },
        ],
        model=variant_medium,
    )

    context_matrix = context.matrix()
    assert isinstance(context_matrix, NDArray)

    context_lookup = context.lookup()
    assert isinstance(context_lookup, dict)
    assert all(isinstance(feature, Feature) for feature in context_lookup.values())

    context_token_activations = context.tokens[-3].inspect()
    assert isinstance(context_token_activations, FeatureActivations)
    assert isinstance(context_token_activations.vector(), NDArray)

    context_token_activations_lookup = context_token_activations.lookup()
    assert isinstance(context_token_activations_lookup, dict)
    assert all(
        isinstance(feature, Feature)
        for feature in context_token_activations_lookup.values()
    )

    top_3_features_activations = context.top(k=3)
    top_1_from_top_3_activations = top_3_features_activations[0]

    assert len(top_3_features_activations) == 3

    assert isinstance(top_3_features_activations, FeatureActivations)
    assert isinstance(top_3_features_activations.vector(), NDArray)

    top_5_features_activations = context.top(k=5)
    top_1_from_top_5_activations = top_5_features_activations[0]

    assert len(top_5_features_activations) == 5

    assert (
        top_1_from_top_3_activations.feature.uuid
        == top_1_from_top_5_activations.feature.uuid
    )
    assert (
        top_1_from_top_3_activations.activation
        == top_1_from_top_5_activations.activation
    )

    top_3_features_activations_lookup = top_3_features_activations.lookup()

    assert isinstance(top_3_features_activations_lookup, dict)
    assert len(top_3_features_activations_lookup) == 3
    assert all(
        isinstance(feature, Feature)
        for feature in top_3_features_activations_lookup.values()
    )

    top_3_features_activations_lookup_keys = [
        int(key) for key in top_3_features_activations_lookup.keys()
    ]

    features_lookup_result = goodfire_client.features.lookup(
        indices=top_3_features_activations_lookup_keys, model=variant_medium
    )

    assert isinstance(features_lookup_result, dict)
    assert len(features_lookup_result) == 3
    assert all(
        isinstance(feature, Feature) for feature in features_lookup_result.values()
    )

    assert (
        features_lookup_result[top_3_features_activations_lookup_keys[0]].uuid
        == top_3_features_activations_lookup[
            top_3_features_activations_lookup_keys[0]
        ].uuid
    )
    assert (
        features_lookup_result[top_3_features_activations_lookup_keys[1]].uuid
        == top_3_features_activations_lookup[
            top_3_features_activations_lookup_keys[1]
        ].uuid
    )
    assert (
        features_lookup_result[top_3_features_activations_lookup_keys[2]].uuid
        == top_3_features_activations_lookup[
            top_3_features_activations_lookup_keys[2]
        ].uuid
    )


def test_workflow_features_insepct_context_object_with_specfic_features_70b(
    goodfire_client, variant_medium
):
    context = goodfire_client.features.inspect(
        [
            {
                "role": "user",
                "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
            },
            {
                "role": "assistant",
                "content": "An example of a synthesizer company is Moog.",
            },
        ],
        model=variant_medium,
    )

    top_1_activation_before_inspecting_specific_features = context.top(k=1)

    company_features = goodfire_client.features.search(
        "company", model=variant_medium, top_k=5
    )

    context_with_specific_features = goodfire_client.features.inspect(
        [
            {
                "role": "user",
                "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
            },
            {
                "role": "assistant",
                "content": "An example of a synthesizer company is Moog.",
            },
        ],
        model=variant_medium,
        features=company_features,
    )

    top_1_activation_after_inspecting_specific_features = (
        context_with_specific_features.top(k=1)
    )

    assert (
        top_1_activation_after_inspecting_specific_features[0].feature.uuid
        != top_1_activation_before_inspecting_specific_features[0].feature.uuid
    )

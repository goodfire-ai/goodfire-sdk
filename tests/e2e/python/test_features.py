import pytest
from goodfire.api.chat.interfaces import ChatMessage
from numpy import ndarray as NDArray

from sdk.python.goodfire.api.exceptions import RequestFailedException
from sdk.python.goodfire.features.features import Feature, FeatureGroup


def test_features_search_base_case(goodfire_client, variant_small, variant_medium):
    result = goodfire_client.features.search(
        "pirate",
        model=variant_small,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 10, "Expected 10 features if no top_k is provided"
    assert all(isinstance(feature, Feature) for feature in result)

    result = goodfire_client.features.search(
        "pirate",
        model=variant_medium,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 10, "Expected 10 features if no top_k is provided"
    assert all(isinstance(feature, Feature) for feature in result)


def test_features_search_with_top_k(goodfire_client, variant_small, variant_medium):
    result = goodfire_client.features.search(
        "pirate",
        model=variant_small,
        top_k=5,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 5, "Expected 5 features if top_k is provided"
    assert all(isinstance(feature, Feature) for feature in result)

    result = goodfire_client.features.search(
        "pirate",
        model=variant_medium,
        top_k=5,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 5, "Expected 5 features if top_k is provided"
    assert all(isinstance(feature, Feature) for feature in result)


def test_features_search_raises_when_top_k_is_out_of_range(
    goodfire_client, variant_small
):
    expected_message_over_upper_bound = "Input should be less than or equal to 100"
    expected_message_under_lower_bound = "Input should be greater than or equal to 1"

    with pytest.raises(RequestFailedException, match=expected_message_over_upper_bound):
        goodfire_client.features.search(
            "pirate",
            model=variant_small,
            top_k=101,
        )

    with pytest.raises(
        RequestFailedException, match=expected_message_under_lower_bound
    ):
        goodfire_client.features.search(
            "pirate",
            model=variant_small,
            top_k=0,
        )


def test_features_search_raises_when_unsupported_model_is_provided(goodfire_client):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.search(
            "pirate",
            model="unsupported-model",
        )


def test_features_neighbors_base_case(goodfire_client, variant_small, variant_medium):
    feature_group = goodfire_client.features.search(
        "pirate",
        model=variant_small,
    )

    selected_feature = feature_group[0]

    result = goodfire_client.features.neighbors(
        selected_feature,
        model=variant_small,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 10  # Expected 10 neighbors if no top_k is provided
    assert all(isinstance(feature, Feature) for feature in result)

    feature_group = goodfire_client.features.search(
        "pirate",
        model=variant_medium,
    )

    selected_feature = feature_group[0]

    result = goodfire_client.features.neighbors(
        selected_feature,
        model=variant_medium,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 10  # Expected 10 neighbors if no top_k is provided
    assert all(isinstance(feature, Feature) for feature in result)


def test_features_neighbors_raises_when_unsupported_model_is_provided(goodfire_client):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.neighbors(
            Feature("test_feature_id", "test_feature_label", 0),
            model="unsupported-model",
        )


def test_features_neighbors_raises_when_number_of_feature_indices_is_out_of_range(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 2048 items after validation, not 2049"
    out_of_range_feature_indices = [
        Feature("test_feature_id", "test_feature_label", 0) for _ in range(2049)
    ]

    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.neighbors(
            out_of_range_feature_indices,
            model=variant_small,
        )

    expected_message = "List should have at least 1 item after validation, not 0"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.neighbors(
            [],
            model=variant_small,
        )


def test_features_neighbors_raises_when_top_k_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "Input should be greater than or equal to 1"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.neighbors(
            [Feature("test_feature_id", "test_feature_label", 0)],
            model=variant_small,
            top_k=0,
        )

    expected_message = "Input should be less than or equal to 8192"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.neighbors(
            [Feature("test_feature_id", "test_feature_label", 0)],
            model=variant_small,
            top_k=8193,
        )


def test_features_rerank_base_case(goodfire_client, variant_small, variant_medium):
    feature_group = goodfire_client.features.search(
        "pirate",
        model=variant_small,
        top_k=5,
    )

    result = goodfire_client.features.rerank(
        feature_group,
        "rum",
        model=variant_small,
        top_k=5,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 5
    assert all(isinstance(feature, Feature) for feature in result)

    feature_group = goodfire_client.features.search(
        "pirate",
        model=variant_medium,
        top_k=5,
    )

    result = goodfire_client.features.rerank(
        feature_group,
        "rum",
        model=variant_medium,
        top_k=5,
    )

    assert isinstance(result, FeatureGroup)
    assert len(result) == 5
    assert all(isinstance(feature, Feature) for feature in result)


def test_features_rerank_raises_when_top_k_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "Input should be greater than or equal to 1"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.rerank(
            FeatureGroup([]),
            "rum",
            model=variant_small,
            top_k=0,
        )

    expected_message = "Input should be less than or equal to 8192"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.rerank(
            FeatureGroup([]),
            "rum",
            model=variant_small,
            top_k=8193,
        )


def test_features_rerank_raises_when_unsupported_model_is_provided(goodfire_client):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.rerank(
            FeatureGroup([]),
            "rum",
            model="unsupported-model",
        )


def test_features_rerank_raises_when_number_of_feature_ids_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 8192 items after validation, not 8193"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.rerank(
            FeatureGroup(
                [
                    Feature("test_feature_id", "test_feature_label", 0)
                    for _ in range(8193)
                ]
            ),
            "rum",
            model=variant_small,
        )


def test_features_activations_base_case(goodfire_client, variant_small, variant_medium):
    result = goodfire_client.features.activations(
        [ChatMessage(role="user", content="What is the capital of France?")],
        model=variant_small,
    )

    assert isinstance(result, NDArray)

    result = goodfire_client.features.activations(
        [ChatMessage(role="user", content="What is the capital of France?")],
        model=variant_medium,
    )

    assert isinstance(result, NDArray)


def test_features_activations_raises_when_list_of_messages_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 128 items after validation, not 129"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.activations(
            [
                ChatMessage(role="user", content="What is the capital of France?")
                for _ in range(129)
            ],
            model=variant_small,
        )

    expected_message = "List should have at least 1 item after validation, not 0"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.activations(
            [],
            model=variant_small,
        )


def test_features_activations_raises_when_list_of_features_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 512 items after validation, not 513"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.activations(
            messages=[
                ChatMessage(role="user", content="What is the capital of France?")
            ],
            model=variant_small,
            features=[
                Feature("test_feature_id", "test_feature_label", 0) for _ in range(513)
            ],
        )


def test_features_activations_raises_when_unsupported_model_is_provided(
    goodfire_client,
):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.activations(
            messages=[
                ChatMessage(role="user", content="What is the capital of France?")
            ],
            model="unsupported-model",
        )


def test_features_inspect_raises_when_number_of_messages_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 128 items after validation, not 129"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.inspect(
            [
                ChatMessage(role="user", content="What is the capital of France?")
                for _ in range(129)
            ],
            model=variant_small,
        )

    expected_message = "List should have at least 1 item after validation, not 0"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.inspect(
            [],
            model=variant_small,
        )


def test_features_inspect_raises_when_number_of_features_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 512 items after validation, not 513"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.inspect(
            [ChatMessage(role="user", content="What is the capital of France?")],
            model=variant_small,
            features=[
                Feature("test_feature_id", "test_feature_label", 0) for _ in range(513)
            ],
        )


def test_features_contrast_base_case(goodfire_client, variant_small, variant_medium):
    result = goodfire_client.features.contrast(
        dataset_1=[
            [
                {"role": "user", "content": "Hello how are you?"},
                {
                    "role": "assistant",
                    "content": "I am a helpful assistant. How can I help you?",
                },
            ]
        ],
        dataset_2=[
            [
                {"role": "user", "content": "Hello how are you?"},
                {
                    "role": "user",
                    "content": "What do you call an alligator in a vest? An investigator.",
                },
            ],
        ],
        model=variant_small,
        top_k=2,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], FeatureGroup)
    assert isinstance(result[1], FeatureGroup)
    assert len(result[0]) == 2
    assert len(result[1]) == 2

    result = goodfire_client.features.contrast(
        dataset_1=[
            [
                {"role": "user", "content": "Hello how are you?"},
                {
                    "role": "assistant",
                    "content": "I am a helpful assistant. How can I help you?",
                },
            ]
        ],
        dataset_2=[
            [
                {"role": "user", "content": "Hello how are you?"},
                {
                    "role": "user",
                    "content": "What do you call an alligator in a vest? An investigator.",
                },
            ],
        ],
        model=variant_medium,
        top_k=2,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], FeatureGroup)
    assert isinstance(result[1], FeatureGroup)
    assert len(result[0]) == 2
    assert len(result[1]) == 2


def test_features_contrast_raises_when_datasets_are_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "List should have at most 64 items after validation, not 65"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.contrast(
            dataset_1=[
                [ChatMessage(role="user", content="Hello how are you?")]
                for _ in range(65)
            ],
            dataset_2=[
                [ChatMessage(role="user", content="Hello how are you?")]
                for _ in range(65)
            ],
            model=variant_small,
        )

    with pytest.raises(ValueError):
        goodfire_client.features.contrast(
            dataset_1=[],
            dataset_2=[],
            model=variant_small,
        )


def test_features_contrast_raises_when_top_k_is_out_of_bounds(
    goodfire_client, variant_small
):
    expected_message = "Input should be greater than or equal to 1"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.contrast(
            dataset_1=[[ChatMessage(role="user", content="Hello how are you?")]],
            dataset_2=[[ChatMessage(role="user", content="Hello how are you?")]],
            model=variant_small,
            top_k=0,
        )

    expected_message = "Input should be less than or equal to 8192"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.contrast(
            dataset_1=[[ChatMessage(role="user", content="Hello how are you?")]],
            dataset_2=[[ChatMessage(role="user", content="Hello how are you?")]],
            model=variant_small,
            top_k=8193,
        )


def test_features_contrast_raises_when_unsupported_model_is_provided(goodfire_client):
    expected_message = "Input should be 'meta-llama/Meta-Llama-3.1-8B-Instruct' or 'meta-llama/Llama-3.3-70B-Instruct'"
    with pytest.raises(RequestFailedException, match=expected_message):
        goodfire_client.features.contrast(
            dataset_1=[[ChatMessage(role="user", content="Hello how are you?")]],
            dataset_2=[[ChatMessage(role="user", content="Hello how are you?")]],
            model="unsupported-model",
        )


def test_features_lookup_base_case(goodfire_client, variant_small, variant_medium):
    result = goodfire_client.features.lookup(
        indices=[0, 1, 2],
        model=variant_small,
    )

    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], Feature)

    result = goodfire_client.features.lookup(
        indices=[0, 1, 2],
        model=variant_medium,
    )

    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], Feature)

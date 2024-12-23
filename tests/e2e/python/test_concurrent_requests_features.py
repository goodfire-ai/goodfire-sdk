import concurrent.futures
import time
from itertools import cycle

import pytest
from goodfire.api.chat.interfaces import ChatMessage

from sdk.tests.e2e.python.config import TestConfig

CONCURRENT_REQUESTS = int(TestConfig.E2E_CONCURRENT_REQUESTS)
ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME = 16


pytest.skip(
    allow_module_level=True,
    reason="Test run on dev keeps running into connection errors",
)


def _search(goodfire_client, variant):
    response = goodfire_client.features.search(
        "music performance",
        model=variant,
    )
    return response


def test_features_search_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_search, goodfire_client, variant_small)
            for _ in range(CONCURRENT_REQUESTS)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent search requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_search_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_search, goodfire_client, variant_medium)
            for _ in range(CONCURRENT_REQUESTS)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent search requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def _neighbors(goodfire_client, variant, test_feature):
    response = goodfire_client.features.neighbors(
        features=test_feature,
        model=variant,
    )
    return response


def test_features_neighbors_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_small,
    )

    test_feature = feature_group[0]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_neighbors, goodfire_client, variant_small, test_feature)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent neighbors requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_neighbors_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_medium,
    )

    test_feature = feature_group[0]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_neighbors, goodfire_client, variant_medium, test_feature)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent neighbors requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def _rerank(goodfire_client, variant, test_feature_group):
    response = goodfire_client.features.rerank(
        features=test_feature_group,
        query="music performance",
        model=variant,
    )
    return response


def test_features_rerank_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    test_feature_group = goodfire_client.features.search(
        "music",
        model=variant_small,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_rerank, goodfire_client, variant_small, test_feature_group)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent rerank requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_rerank_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    test_feature_group = goodfire_client.features.search(
        "music",
        model=variant_medium,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(
                _rerank, goodfire_client, variant_medium, test_feature_group
            )
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent rerank requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def _activations(goodfire_client, variant, test_feature_group):
    response = goodfire_client.features.activations(
        messages=[
            ChatMessage(
                role="user",
                content="What kind of performances happen at the Walt Disney Concert Hall?",
            ),
            ChatMessage(
                role="assistant",
                content="The Walt Disney Concert Hall is a performing arts center in Los Angeles, California, that hosts a variety of concerts, operas, and other performances. The hall is known for its stunning architecture and acoustics, making it a popular venue for both local and international artists.",
            ),
        ],
        model=variant,
        features=test_feature_group,
    )
    return response


def test_features_activations_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    test_feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_small,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(
                _activations, goodfire_client, variant_small, test_feature_group
            )
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent activations requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_activations_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    test_feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_medium,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(
                _activations, goodfire_client, variant_medium, test_feature_group
            )
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent activations requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def _inspect(goodfire_client, variant, test_feature_group, aggregate_by):
    response = goodfire_client.features.inspect(
        messages=[
            ChatMessage(
                role="user",
                content="What kind of performances happen at the Walt Disney Concert Hall?",
            ),
            ChatMessage(
                role="assistant",
                content="The Walt Disney Concert Hall is a performing arts center in Los Angeles, California, that hosts a variety of concerts, operas, and other performances. The hall is known for its stunning architecture and acoustics, making it a popular venue for both local and international artists.",
            ),
        ],
        model=variant,
        features=test_feature_group,
        aggregate_by=aggregate_by,
    )
    return response


aggregate_by = ["frequency", "mean", "max", "sum"]
aggregate_by_cycle = cycle(aggregate_by)


def test_features_inspect_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    test_feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_small,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(
                _inspect,
                goodfire_client,
                variant_small,
                test_feature_group,
                next(aggregate_by_cycle),
            )
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent inspect requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_inspect_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    test_feature_group = goodfire_client.features.search(
        "music performance",
        model=variant_medium,
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(
                _inspect,
                goodfire_client,
                variant_medium,
                test_feature_group,
                next(aggregate_by_cycle),
            )
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent inspect requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def _contrast(goodfire_client, variant):
    response = goodfire_client.features.contrast(
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
        model=variant,
    )
    return response


def test_features_contrast_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_contrast, goodfire_client, variant_small)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent contrast requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_contrast_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_contrast, goodfire_client, variant_medium)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent contrast requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def _lookup(goodfire_client, variant):
    response = goodfire_client.features.lookup(
        indices=[48499, 55152],
        model=variant,
    )
    return response


def test_features_lookup_concurrent_requests_baseline_8b(
    goodfire_client, variant_small
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_lookup, goodfire_client, variant_small)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent lookup requests for {variant_small.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME


def test_features_lookup_concurrent_requests_baseline_70b(
    goodfire_client, variant_medium
):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        start_time = time.time()

        futures = [
            executor.submit(_lookup, goodfire_client, variant_medium)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"Time taken for {CONCURRENT_REQUESTS} concurrent lookup requests for {variant_medium.base_model}: {total_time:.2f} seconds"
        )

        assert total_time < ACCEPTABLE_CONCURRENT_REQUESTS_COMPLETION_TIME

def test_workflow_features_contrast_rerank_8b(goodfire_client, variant_small):
    dataset_1_features, dataset_2_features = goodfire_client.features.contrast(
        dataset_1=[
            [
                {
                    "role": "user",
                    "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
                },
                {
                    "role": "assistant",
                    "content": "An example of a synthesizer company is Moog.",
                },
            ]
        ],
        dataset_2=[
            [
                {"role": "user", "content": "How do synthesizers work?"},
                {
                    "role": "assistant",
                    "content": "Synthesizers are electronic instruments that generate sound through various techniques, including analog and digital synthesis.",
                },
            ],
        ],
        model=variant_small,
        top_k=10,
    )

    top_activating_feature_from_dataset_2_before_rerank = dataset_2_features[0].uuid

    instruments_features = goodfire_client.features.rerank(
        dataset_2_features, "instruments", model=variant_small, top_k=5
    )

    top_activating_feature_from_dataset_2_after_rerank = instruments_features[0].uuid

    assert (
        top_activating_feature_from_dataset_2_after_rerank
        != top_activating_feature_from_dataset_2_before_rerank
    )


def test_workflow_features_contrast_rerank_70b(goodfire_client, variant_medium):
    dataset_1_features, dataset_2_features = goodfire_client.features.contrast(
        dataset_1=[
            [
                {
                    "role": "user",
                    "content": "Analog synthesizers are interesting instruments. What is an example of a synthesizer company?",
                },
                {
                    "role": "assistant",
                    "content": "An example of a synthesizer company is Moog.",
                },
            ]
        ],
        dataset_2=[
            [
                {"role": "user", "content": "How do synthesizers work?"},
                {
                    "role": "assistant",
                    "content": "Synthesizers are electronic instruments that generate sound through various techniques, including analog and digital synthesis.",
                },
            ],
        ],
        model=variant_medium,
        top_k=10,
    )

    top_activating_feature_from_dataset_2_before_rerank = dataset_2_features[0].uuid

    instruments_features = goodfire_client.features.rerank(
        dataset_2_features, "instruments", model=variant_medium, top_k=5
    )

    top_activating_feature_from_dataset_2_after_rerank = instruments_features[0].uuid

    assert (
        top_activating_feature_from_dataset_2_after_rerank
        != top_activating_feature_from_dataset_2_before_rerank
    )

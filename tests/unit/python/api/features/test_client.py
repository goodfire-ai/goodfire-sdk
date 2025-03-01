from unittest.mock import patch, AsyncMock, MagicMock
from uuid import UUID

from goodfire.api.features.client import FeaturesAPI
from goodfire.features.features import Feature

MOCK_BASE_URL = "https://api.goodfire.ai"


def test_search():
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = """
            {
                "features": [
                    {
                        "id": "00000000-0000-0000-0000-000000000000",
                        "label": "feature",
                        "max_activation_strength": 0.5,
                        "index_in_sae": 1,
                        "relevance": 0.8
                    }
                ]
            }
        """

        client = FeaturesAPI("api_key", base_url=MOCK_BASE_URL)
        response = client.search("query", "model", top_k=10)

        mock_get.assert_called_once_with(
            f"{MOCK_BASE_URL}/api/inference/v1/features/search",
            params={"query": "query", "page": 1, "perPage": 10, "model": "model"},
            headers={
                "Authorization": "Bearer api_key",
                "Content-Type": "application/json",
                "X-Base-Url": MOCK_BASE_URL,
            },
            timeout=10,
        )

        assert response._features[0].uuid == UUID(
            "00000000-0000-0000-0000-000000000000"
        )
        assert response._features[0].label == "feature"
        assert response._features[0].index_in_sae == 1


# def test_list():
#     with patch("requests.get") as mock_get:
#         mock_get.return_value.status_code = 200
#         mock_get.return_value.text = """
#             {
#                 "features": [
#                     {
#                         "id": "00000000-0000-0000-0000-000000000000",
#                         "label": "feature",
#                         "max_activation_strength": 0.5,
#                         "index_in_sae": 1
#                     },
#                     {
#                         "id": "11111111-1111-1111-1111-111111111111",
#                         "label": "another feature",
#                         "max_activation_strength": 0.6,
#                         "index_in_sae": 2
#                     }
#                 ]
#             }
#         """

#         client = FeaturesAPI("api_key")
#         response = client.list(
#             [
#                 "00000000-0000-0000-0000-000000000000",
#                 "11111111-1111-1111-1111-111111111111",
#             ]
#         )

#         mock_get.assert_called_once_with(
#             "https://api.goodfire.ai/api/inference/v1/features",
#             params={
#                 "ids": "00000000-0000-0000-0000-000000000000,11111111-1111-1111-1111-111111111111"
#             },
#             headers={
#                 "Authorization": "Bearer api_key",
#                 "Content-Type": "application/json",
#             },
#         )

#         uuid1 = UUID("00000000-0000-0000-0000-000000000000")
#         assert response._features[uuid1].uuid == uuid1
#         assert response._features[uuid1].label == "feature"
#         assert response._features[uuid1].max_activation_strength == 0.5
#         assert response._features[uuid1].index_in_sae == 1

#         uuid2 = UUID("11111111-1111-1111-1111-111111111111")
#         assert response._features[uuid2].uuid == uuid2
#         assert response._features[uuid2].label == "another feature"
#         assert response._features[uuid2].max_activation_strength == 0.6
#         assert response._features[uuid2].index_in_sae == 2


# def test_get():
#     with patch("requests.get") as mock_get:
#         mock_get.return_value.status_code = 200
#         mock_get.return_value.text = """
#             {
#                 "features": [
#                     {
#                         "id": "00000000-0000-0000-0000-000000000000",
#                         "label": "feature",
#                         "max_activation_strength": 0.5,
#                         "index_in_sae": 1
#                     }
#                 ]
#             }
#         """

#         client = FeaturesAPI("api_key")
#         response = client.get("00000000-0000-0000-0000-000000000000")

#         mock_get.assert_called_once_with(
#             "https://api.goodfire.ai/api/inference/v1/features",
#             params={"ids": "00000000-0000-0000-0000-000000000000"},
#             headers={
#                 "Authorization": "Bearer api_key",
#                 "Content-Type": "application/json",
#             },
#         )

#         uuid = UUID("00000000-0000-0000-0000-000000000000")
#         assert response.uuid == uuid
#         assert response.label == "feature"
#         assert response.max_activation_strength == 0.5
#         assert response.index_in_sae == 1


def test_attribute():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Create a MagicMock for the response that doesn't return a coroutine for json()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "to_ablate": [
                {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "value": 0.81640625,
                    "tokens": [
                        {"index": 0, "activation_strength": 0.419921875},
                        {"index": 5, "activation_strength": 0.81640625},
                        {"index": 6, "activation_strength": 0.7109375}
                    ]
                },
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "value": 1.03125,
                    "tokens": [
                        {"index": 18, "activation_strength": 0.08740234375},
                        {"index": 30, "activation_strength": 1.03125},
                        {"index": 31, "activation_strength": 0.115234375}
                    ]
                }
            ],
            "to_add": [],
            "num_input_tokens": 33
        }
        mock_post.return_value = mock_response

        # Mock the _list method to return feature data
        with patch("goodfire.api.features.client.AsyncFeaturesAPI._list", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [
                Feature(
                    uuid=UUID("00000000-0000-0000-0000-000000000000"),
                    label="feature1",
                    index_in_sae=1,
                ),
                Feature(
                    uuid=UUID("11111111-1111-1111-1111-111111111111"),
                    label="feature2",
                    index_in_sae=2,
                ),
            ]

            client = FeaturesAPI("api_key", base_url=MOCK_BASE_URL)
            messages = [{"role": "user", "content": "Hello"}]
            response = client.attribute(messages, 0, "model")

            # Check API call
            mock_post.assert_called_once_with(
                f"{MOCK_BASE_URL}/api/inference/v1/attributions/compute-logit-attribution",
                headers={
                    "Authorization": "Bearer api_key",
                    "Content-Type": "application/json",
                    "X-Base-Url": MOCK_BASE_URL,
                },
                json={
                    "messages": messages,
                    "model": "model",
                    "startIndex": 0,
                    "endIndex": 0,
                },
                timeout=10,
            )

            # Check feature list call
            mock_list.assert_called_once()
            
            # Check response structure
            assert len(response.features) == 2
            assert response.num_input_tokens == 33
            
            # Check first feature (should be sorted by value)
            assert response.features[0].feature_id == "11111111-1111-1111-1111-111111111111"
            assert response.features[0].value == 1.03125
            assert len(response.features[0].token_activations) == 3
            assert response.features[0].token_activations[0].index == 18
            assert response.features[0].token_activations[0].activation_strength == 0.08740234375
            assert response.features[0].feature.label == "feature2"
            
            # Check second feature
            assert response.features[1].feature_id == "00000000-0000-0000-0000-000000000000"
            assert response.features[1].value == 0.81640625
            assert len(response.features[1].token_activations) == 3
            assert response.features[1].token_activations[0].index == 0
            assert response.features[1].token_activations[0].activation_strength == 0.419921875
            assert response.features[1].feature.label == "feature1"


def test_attribute_without_feature_data():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Create a MagicMock for the response that doesn't return a coroutine for json()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "to_ablate": [
                {
                    "id": "00000000-0000-0000-0000-000000000000",
                    "value": 0.81640625,
                    "tokens": [
                        {"index": 0, "activation_strength": 0.419921875},
                        {"index": 5, "activation_strength": 0.81640625}
                    ]
                }
            ],
            "to_add": [],
            "num_input_tokens": 10
        }
        mock_post.return_value = mock_response

        # Mock the _list method to verify it's not called
        with patch("goodfire.api.features.client.AsyncFeaturesAPI._list", new_callable=AsyncMock) as mock_list:
            client = FeaturesAPI("api_key", base_url=MOCK_BASE_URL)
            messages = [{"role": "user", "content": "Hello"}]
            response = client.attribute(messages, 0, "model", _fetch_feature_data=False)

            # Check API call
            mock_post.assert_called_once()
            
            # Verify _list was not called
            mock_list.assert_not_called()
            
            # Check response structure
            assert len(response.features) == 1
            assert response.num_input_tokens == 10
            
            # Check feature
            assert response.features[0].feature_id == "00000000-0000-0000-0000-000000000000"
            assert response.features[0].value == 0.81640625
            assert len(response.features[0].token_activations) == 2
            assert response.features[0].feature is None

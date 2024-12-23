from unittest.mock import patch
from uuid import UUID

from sdk.python.goodfire.api.features.client import FeaturesAPI

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

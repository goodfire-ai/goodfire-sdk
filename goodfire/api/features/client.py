from typing import Any, Iterable, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ...features.features import ConditionalGroup, Feature, FeatureEdits, FeatureGroup
from ...utils.asyncio import run_async_safely
from ...variants.variants import SUPPORTED_MODELS, Variant
from ..chat.interfaces import ChatMessage
from ..constants import PRODUCTION_BASE_URL
from ..utils import AsyncHTTPWrapper
from .interfaces import SearchFeatureResponse


class AsyncFeaturesAPI:
    """A class for accessing interpretable SAE features of AI models."""

    def __init__(
        self,
        goodfire_api_key: str,
        base_url: str = PRODUCTION_BASE_URL,
    ):
        self.goodfire_api_key = goodfire_api_key
        self.base_url = base_url

        self._http = AsyncHTTPWrapper()

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.goodfire_api_key}",
            "Content-Type": "application/json",
            "X-Base-Url": self.base_url,
        }

    async def neighbors(
        self,
        features: Union[Feature, FeatureGroup],
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 10,
    ) -> FeatureGroup:
        """Get the nearest neighbors of a feature or group of features."""
        if isinstance(features, Feature):
            features = FeatureGroup([features])

        url = f"{self.base_url}/api/inference/v1/attributions/neighbors"
        payload = {
            "feature_indices": [feature.index_in_sae for feature in features],
            "model": model if isinstance(model, str) else model.base_model,
            "top_k": top_k,
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers)

        response_body = response.json()

        results: list[Feature] = []
        for feature in response_body["neighbors"]:
            results.append(
                Feature(
                    uuid=feature["id"],
                    label=feature["label"],
                    index_in_sae=feature["index_in_sae"],
                )
            )

        return FeatureGroup(results)

    async def search(
        self,
        query: str,
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 10,
    ) -> FeatureGroup:
        """Search for features based on a query."""
        url = f"{self.base_url}/api/inference/v1/features/search"
        params = {
            "query": query,
            "page": 1,
            "perPage": top_k,
            "model": model if isinstance(model, str) else model.base_model,
        }
        headers = self._get_headers()
        response = await self._http.get(url, params=params, headers=headers)

        response = SearchFeatureResponse.model_validate_json(response.text)

        features: list[Feature] = []
        for feature in response.features:
            features.append(
                Feature(
                    uuid=feature.id,
                    label=feature.label,
                    index_in_sae=feature.index_in_sae,
                )
            )

        return FeatureGroup(features)

    async def semantic_similarity(
        self,
        features: FeatureGroup,
        query: str,
        model: Union[SUPPORTED_MODELS, Variant],
    ) -> list[float]:
        url = f"{self.base_url}/api/inference/v1/features/rerank"
        payload = {
            "query": query,
            "top_k": len(features),
            "model": model if isinstance(model, str) else model.base_model,
            "feature_ids": [str(feature.uuid) for feature in features],
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers)

        response = SearchFeatureResponse.model_validate_json(response.text)

        semantic_similarity_scores: list[float] = []
        for feature in response.features:
            semantic_similarity_scores.append(feature.relevance)

        return semantic_similarity_scores

    async def rerank(
        self,
        features: FeatureGroup,
        query: str,
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 10,
    ) -> FeatureGroup:
        """Rerank a set of features based on a query."""
        url = f"{self.base_url}/api/inference/v1/features/rerank"
        payload = {
            "query": query,
            "top_k": top_k,
            "model": model if isinstance(model, str) else model.base_model,
            "feature_ids": [str(feature.uuid) for feature in features],
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers, timeout=30)

        response = SearchFeatureResponse.model_validate_json(response.text)

        features_to_return: list[Feature] = []
        for feature in response.features:
            features_to_return.append(
                Feature(
                    uuid=feature.id,
                    label=feature.label,
                    index_in_sae=feature.index_in_sae,
                )
            )

        return FeatureGroup(features_to_return)

    async def activations(
        self,
        messages: list[ChatMessage],
        model: Union[SUPPORTED_MODELS, Variant],
        features: Optional[Union[Feature, FeatureGroup]] = None,
    ) -> NDArray[np.float64]:
        """Retrieve feature activations matrix for a set of messages."""

        context = await self.inspect(
            messages, model, features, _fetch_feature_data=False
        )

        return context.matrix()

    async def inspect(
        self,
        messages: list[ChatMessage],
        model: Union[SUPPORTED_MODELS, Variant],
        features: Optional[Union[Feature, FeatureGroup]] = None,
        aggregate_by: Literal["frequency", "mean", "max", "sum"] = "frequency",
        _fetch_feature_data: bool = True,
    ):
        """Inspect feature activations for a set of messages."""
        payload: dict[str, Any] = {
            "messages": messages,
            "aggregate_by": "count" if aggregate_by == "frequency" else aggregate_by,
        }

        if isinstance(model, str):
            payload["model"] = model
        else:
            payload["model"] = model.base_model

            payload["controller"] = model.controller.json()

        include_feature_ids: Optional[set[str]] = None
        if features:
            if isinstance(features, Feature):
                include_feature_indexes = [features.index_in_sae]
                include_feature_ids = {str(features.uuid)}
            else:
                include_feature_indexes: list[int] = []
                include_feature_ids = set()
                for f in features:
                    include_feature_ids.add(str(f.uuid))
                    include_feature_indexes.append(f.index_in_sae)

            payload["include_feature_indexes"] = include_feature_indexes

        response = await self._http.post(
            f"{self.base_url}/api/inference/v1/chat-attribution/compute-features",
            headers=self._get_headers(),
            json=payload,
            timeout=30,
        )

        inspector = ContextInspector(
            self,
            response.json(),
            model=model,
            aggregate_by=aggregate_by,
            include_feature_ids=include_feature_ids,
        )

        if _fetch_feature_data:
            await inspector.fetch_features()

        return inspector

    async def _tokenize(
        self,
        messages: list[ChatMessage],
        model: Union[SUPPORTED_MODELS, Variant],
    ):
        """Tokenize messages."""
        payload: dict[str, Any] = {
            "messages": messages,
            "model": model if isinstance(model, str) else model.base_model,
        }

        response = await self._http.post(
            f"{self.base_url}/api/inference/v1/chat/tokenize",
            headers=self._get_headers(),
            json=payload,
        )

        return [token["value"] for token in response.json()["tokens"]]

    async def contrast(
        self,
        dataset_1: list[list[ChatMessage]],
        dataset_2: list[list[ChatMessage]],
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 5,
    ) -> tuple[FeatureGroup, FeatureGroup]:
        """Identify features that differentiate between two conversation datasets.

        Args:
            dataset_1: First conversation dataset
            dataset_2: Second conversation dataset
            model: Model identifier or variant interface
            top_k: Number of top features to return (default: 5)

        Returns:
            tuple: Two FeatureGroups containing:
                - Features steering towards dataset_1
                - Features steering towards dataset_2

            Each Feature has properties:
                - uuid: Unique feature identifier
                - label: Human-readable feature description
                - index_in_sae: Index in sparse autoencoder

        Raises:
            ValueError: If datasets are empty or have different lengths

        Example:
            >>> dataset_1 = [[
            ...     {"role": "user", "content": "Hi how are you?"},
            ...     {"role": "assistant", "content": "I'm doing well..."}
            ... ]]
            >>> dataset_2 = [[
            ...     {"role": "user", "content": "Hi how are you?"},
            ...     {"role": "assistant", "content": "Arr my spirits be high..."}
            ... ]]
            >>> features_1, features_2 = client.features.contrast(
            ...     dataset_1=dataset_1,
            ...     dataset_2=dataset_2,
            ...     model=model,
            ...     dataset_2_feature_rerank_query="pirate",
            ...     top_k=5
            ... )
        """
        if len(dataset_1) != len(dataset_2):
            raise ValueError("dataset_1 and dataset_2 must have the same length")

        if len(dataset_1) == 0:
            raise ValueError("dataset_1 and dataset_2 must have at least one element")

        url = f"{self.base_url}/api/inference/v1/chat-attribution/contrast"
        payload = {
            "dataset_1": dataset_1,
            "dataset_2": dataset_2,
            "k_to_add": top_k,
            "k_to_remove": top_k,
            "model": model if isinstance(model, str) else model.base_model,
        }

        headers = self._get_headers()
        response = await self._http.post(
            url, json=payload, headers=headers, timeout=120
        )

        response_body = response.json()

        dataset_1_features = FeatureGroup(
            [
                Feature(
                    uuid=feature["id"],
                    label=feature["label"],
                    index_in_sae=feature["index_in_sae"],
                )
                for feature in response_body["dataset_1_features"]
            ]
        )
        dataset_2_features = FeatureGroup(
            [
                Feature(
                    uuid=feature["id"],
                    label=feature["label"],
                    index_in_sae=feature["index_in_sae"],
                )
                for feature in response_body["dataset_2_features"]
            ]
        )

        return dataset_1_features, dataset_2_features

    async def lookup(
        self,
        indices: list[int],
        model: Union[SUPPORTED_MODELS, Variant],
    ) -> dict[int, Feature]:
        url = f"{self.base_url}/api/inference/v1/features/lookup"
        payload = {
            "indices": indices,
            "model": model if isinstance(model, str) else model.base_model,
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers)

        json_response = response.json()
        features = json_response["features"]

        lookup_response = {}
        for feature in features:
            lookup_response[feature["index_in_sae"]] = Feature(
                uuid=feature["uuid"],
                label=feature["label"],
                index_in_sae=feature["index_in_sae"],
            )

        return lookup_response

    async def generate_contrastive_stimulus(
        self,
        specification: str,
    ):
        url = f"{self.base_url}/api/inference/v1/attributions/generate-contrastive-dataset"
        payload = {
            "specification": specification,
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers, timeout=30)

        return response.json()[0], response.json()[1]

    async def _list(self, ids: "list[str]") -> FeatureGroup:
        """Get features by their IDs."""
        url = f"{self.base_url}/api/inference/v1/features/"
        params = {
            "feature_id": ids,
        }
        headers = self._get_headers()
        response = await self._http.get(url, params=params, headers=headers)

        response = SearchFeatureResponse.model_validate_json(response.text)

        return FeatureGroup(
            [
                Feature(
                    uuid=feature.id,
                    label=feature.label,
                    index_in_sae=feature.index_in_sae,
                )
                for feature in response.features
            ]
        )

    async def AutoSteer(
        self,
        specification: str,
        model: Union[SUPPORTED_MODELS, Variant],
    ):
        url = f"{self.base_url}/api/inference/v1/features/auto-edits"
        payload = {
            "specification": specification,
            "model": model if isinstance(model, str) else model.base_model,
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers, timeout=60)

        return FeatureEdits(
            [
                (
                    Feature(
                        uuid=feature["id"],
                        label=feature["label"],
                        index_in_sae=feature["index_in_sae"],
                    ),
                    strength,
                )
                for feature, strength in response.json()["edits"]
            ]
        )

    async def AutoConditional(
        self,
        specification: str,
        model: Union[SUPPORTED_MODELS, Variant],
    ) -> FeatureGroup:
        url = f"{self.base_url}/api/inference/v1/features/auto-conditional"
        payload = {
            "specification": specification,
            "model": model if isinstance(model, str) else model.base_model,
        }
        headers = self._get_headers()
        response = await self._http.post(url, json=payload, headers=headers, timeout=60)

        return ConditionalGroup.from_json(response.json()["conditional"])

    async def attribute(
        self,
        messages: list[ChatMessage],
        index: int,
        model: Union[SUPPORTED_MODELS, Variant],
        _fetch_feature_data: bool = True,
    ):
        payload = {
            "messages": messages,
            "model": model if isinstance(model, str) else model.base_model,
            "startIndex": index,
            "endIndex": index,
        }

        response = await self._http.post(
            f"{self.base_url}/api/inference/v1/attributions/compute-logit-attribution",
            headers=self._get_headers(),
            json=payload,
        )

        attribution = AttributionResponse(response.json(), self)

        if _fetch_feature_data:
            await attribution.fetch_features()

        return attribution


class FeatureActivation:
    def __init__(self, feature: Feature, activation_strength: float):
        self.feature = feature
        self.activation = activation_strength

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"FeatureActivation(feature={self.feature}, activation={self.activation})"
        )


class FeatureActivations:
    def __init__(
        self,
        acts: Iterable[tuple[Feature, float]],
        model: Union[SUPPORTED_MODELS, Variant],
    ):
        self._acts = [FeatureActivation(feat, act) for feat, act in acts]
        self.model = model

    def __getitem__(self, idx: int):
        return self._acts[idx]

    def __iter__(self):
        return iter(self._acts)

    def __len__(self):
        return len(self._acts)

    def __repr__(self):
        return str(self)

    def __str__(self):
        response_str = "FeatureActivations("

        for index, act in enumerate(self._acts[:10]):
            response_str += f"\n{index}: ({act.feature}, {act.activation})"

        if len(self._acts) > 10:
            response_str += "\n..."
            response_str += f"\n{len(self._acts) - 1}: ({self._acts[-1].feature}, {self._acts[-1].activation})"

        response_str = response_str.replace("\n", "\n   ")

        response_str += "\n)"

        return response_str

    def vector(self):
        model_name = (
            self.model if isinstance(self.model, str) else self.model.base_model
        )
        SAE_SIZE = (
            131072 if model_name == "meta-llama/Llama-3.3-70B-Instruct" else 65536
        )
        array = np.zeros(SAE_SIZE)

        for act in self._acts:
            array[act.feature.index_in_sae] = act.activation

        return array

    def lookup(self) -> dict[int, Feature]:
        return {act.feature.index_in_sae: act.feature for act in self._acts}


class Token:
    def __init__(
        self,
        context: "ContextInspector",
        token: str,
        feature_acts: list[dict[str, Any]],
    ):
        self._token = token
        self._feature_acts = feature_acts
        self._context = context

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'Token("{self._token}")'

    def inspect(self, k: int = 5):
        uuids = [act["id"] for act in self._feature_acts[:k]]
        features = [
            self._context._features[uuid]
            for uuid in uuids
            if uuid in self._context._features
        ]

        return FeatureActivations(
            tuple(
                (feature, act["activation_strength"])
                for feature, act in zip(features, self._feature_acts)
            ),
            model=self._context.model,
        )

    def vector(self) -> NDArray[np.float64]:
        model_name = (
            self._context.model.base_model
            if isinstance(self._context.model, Variant)
            else self._context.model
        )
        SAE_SIZE = (
            131072 if model_name == "meta-llama/Llama-3.3-70B-Instruct" else 65536
        )
        array = np.zeros(SAE_SIZE)

        for act in self._feature_acts:
            array[self._context._feature_indices[act["id"]]] = act[
                "activation_strength"
            ]

        return array

    def lookup(self) -> dict[int, Feature]:
        lookup: dict[int, Feature] = {}
        for act in self._feature_acts:
            if self._context._features.get(act["id"]):
                lookup[self._context._feature_indices[act["id"]]] = (
                    self._context._features[act["id"]]
                )
        return lookup


class ContextInspector:
    def __init__(
        self,
        client: AsyncFeaturesAPI,
        context_response: dict[str, Any],
        model: Union[SUPPORTED_MODELS, Variant],
        aggregate_by: Literal["frequency", "mean", "max", "sum"] = "frequency",
        include_feature_ids: Optional[set[str]] = None,
    ):
        self._client = client
        self.tokens: list[Token] = []
        self._feature_strengths: dict[str, list[float]] = {}
        self._feature_indices: dict[str, int] = {}

        self._feature_ids: set[str] = set()

        self.model = model

        if include_feature_ids:
            for id in include_feature_ids:
                self._feature_strengths[id] = [0, 0]
                self._feature_ids.add(id)

        for token_config in context_response["tokens"]:
            self.tokens.append(
                Token(self, token_config["token"], token_config["attributions"])
            )
            for act in token_config["attributions"]:
                if not self._feature_indices.get(act["id"]):
                    self._feature_indices[act["id"]] = act["index_in_sae"]

                self._feature_ids.add(act["id"])

                if not self._feature_strengths.get(act["id"]):
                    self._feature_strengths[act["id"]] = [0, 0]

                if act["activation_strength"] > 0.1:
                    self._feature_strengths[act["id"]][0] += 1
                    if aggregate_by == "frequency":
                        self._feature_strengths[act["id"]][1] += 1
                    elif aggregate_by == "max":
                        if (
                            act["activation_strength"]
                            > self._feature_strengths[act["id"]][1]
                        ):
                            self._feature_strengths[act["id"]][1] = act[
                                "activation_strength"
                            ]
                    else:
                        self._feature_strengths[act["id"]][1] += act[
                            "activation_strength"
                        ]

        if aggregate_by == "mean":
            for feature_strength in self._feature_strengths.values():
                if feature_strength[0]:
                    feature_strength[1] /= feature_strength[0]

        self._features = {}


    async def fetch_features(self):
        features: list[Feature] = []
        for chunk_start in range(0, len(self._feature_ids), 50):
            features += await self._client._list(
                list(self._feature_ids)[chunk_start : chunk_start + 50]
            )
        self._features: dict[str, Feature] = {str(f.uuid): f for f in features}

    def __repr__(self):
        return str(self)

    def __str__(self):
        response_str = "ContextInspector(\n"

        for token in self.tokens[:50]:
            response_str += f"{token._token}"

        response_str = response_str.replace("\n", "\n   ")

        if len(self.tokens) >= 50:
            response_str += "..."

        response_str += "\n)"

        return response_str

    def top(self, k: int = 5):
        sorted_feature_ids = sorted(
            list(self._feature_strengths.items()),
            key=lambda row: row[1][0],
            reverse=True,
        )

        features = [
            self._features[feat_id]
            for feat_id, _ in sorted_feature_ids[:k]
            if self._features.get(feat_id)
        ]

        return FeatureActivations(
            sorted(
                tuple(
                    (feature, self._feature_strengths[str(feature.uuid)][1])
                    for feature in features
                ),
                key=lambda row: row[1],
                reverse=True,
            ),
            model=self.model,
        )

    def matrix(self):
        token_vectors: list[NDArray[np.float64]] = []
        for token in self.tokens:
            token_vectors.append(token.vector())

        return np.array(token_vectors)

    def lookup(self) -> dict[int, Feature]:
        lookup: dict[int, Feature] = {}
        for token in self.tokens:
            lookup.update(token.lookup())
        return lookup


class TokenActivation:
    def __init__(self, index: int, activation_strength: float):
        self.index = index
        self.activation_strength = activation_strength

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"TokenActivation(index={self.index}, activation={self.activation_strength})"


class FeatureAttribution:
    def __init__(self, feature_id: str, value: float, tokens: list[dict[str, Any]]):
        self.feature_id = feature_id
        self.value = value
        self.token_activations = [
            TokenActivation(token["index"], token["activation_strength"])
            for token in tokens
        ]
        self._feature: Optional[Feature] = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self._feature is None:
            return f"FeatureAttribution(id={self.feature_id}, value={self.value}, tokens={len(self.token_activations)})"
        else:
            return f"FeatureAttribution(feature={self._feature}, value={self.value}, tokens={len(self.token_activations)})"

    @property
    def feature(self) -> Optional[Feature]:
        return self._feature

    @feature.setter
    def feature(self, feature: Feature):
        self._feature = feature


class AttributionResponse:
    def __init__(self, response_data: dict[str, Any], client: AsyncFeaturesAPI):
        self._client = client
        self.features = sorted([
            FeatureAttribution(item["id"], item["value"], item["tokens"])
            for item in response_data.get("to_ablate", [])
        ], key = lambda row: row.value,
        reverse = True)
        self.num_input_tokens = response_data.get("num_input_tokens", 0)
        self._features_loaded = False

    async def fetch_features(self):
        """Fetch feature details for all attributions."""
        if self._features_loaded:
            return

        feature_ids = set()
        for attribution in self.features:
            feature_ids.add(attribution.feature_id)

        if not feature_ids:
            self._features_loaded = True
            return

        features_list = []
        for chunk_start in range(0, len(feature_ids), 50):
            features_list += await self._client._list(
                list(feature_ids)[chunk_start:chunk_start + 50]
            )

        features_by_id = {str(f.uuid): f for f in features_list}

        # Assign features to attributions
        for attribution in self.features:
            if attribution.feature_id in features_by_id:
                attribution.feature = features_by_id[attribution.feature_id]

        self._features_loaded = True

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AttributionResponse(features={self.features}, tokens={self.num_input_tokens})"


class FeaturesAPI:
    def __init__(self, api_key: str, base_url: str = PRODUCTION_BASE_URL):
        self._async_client = AsyncFeaturesAPI(api_key, base_url=base_url)

    def neighbors(
        self,
        features: Union[Feature, FeatureGroup],
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 10,
    ) -> FeatureGroup:
        return run_async_safely(self._async_client.neighbors(features, model, top_k))

    def search(
        self,
        query: str,
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 10,
    ) -> FeatureGroup:
        return run_async_safely(self._async_client.search(query, model, top_k))

    def rerank(
        self,
        features: FeatureGroup,
        query: str,
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 10,
    ) -> FeatureGroup:
        return run_async_safely(
            self._async_client.rerank(features, query, model, top_k)
        )

    def activations(
        self,
        messages: list[ChatMessage],
        model: Union[SUPPORTED_MODELS, Variant],
        features: Optional[Union[Feature, FeatureGroup]] = None,
    ) -> NDArray[np.float64]:
        return run_async_safely(
            self._async_client.activations(messages, model, features)
        )

    def inspect(
        self,
        messages: list[ChatMessage],
        model: Union[SUPPORTED_MODELS, Variant],
        features: Optional[Union[Feature, FeatureGroup]] = None,
        aggregate_by: Literal["frequency", "mean", "max", "sum"] = "frequency",
        _fetch_feature_data: bool = True,
    ):
        return run_async_safely(
            self._async_client.inspect(
                messages, model, features, aggregate_by, _fetch_feature_data
            )
        )

    def contrast(
        self,
        dataset_1: list[list[ChatMessage]],
        dataset_2: list[list[ChatMessage]],
        model: Union[SUPPORTED_MODELS, Variant],
        top_k: int = 5,
    ) -> tuple[FeatureGroup, FeatureGroup]:
        return run_async_safely(
            self._async_client.contrast(
                dataset_1,
                dataset_2,
                model,
                top_k,
            )
        )

    def lookup(
        self,
        indices: list[int],
        model: Union[SUPPORTED_MODELS, Variant],
    ) -> dict[int, Feature]:
        return run_async_safely(self._async_client.lookup(indices, model))

    def generate_contrastive_stimulus(
        self,
        specification: str,
    ):
        return run_async_safely(
            self._async_client.generate_contrastive_stimulus(specification)
        )

    def attribute(
        self,
        messages: list[ChatMessage],
        index: int,
        model: Union[SUPPORTED_MODELS, Variant],
        _fetch_feature_data: bool = True,
    ):
        return run_async_safely(self._async_client.attribute(messages, index, model, _fetch_feature_data))

    def AutoSteer(
        self,
        specification: str,
        model: Union[SUPPORTED_MODELS, Variant],
    ):
        return run_async_safely(self._async_client.AutoSteer(specification, model))

    def AutoConditional(
        self,
        specification: str,
        model: Union[SUPPORTED_MODELS, Variant],
    ) -> ConditionalGroup:
        return run_async_safely(
            self._async_client.AutoConditional(specification, model)
        )

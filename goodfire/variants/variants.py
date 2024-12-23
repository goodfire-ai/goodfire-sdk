import json
import uuid
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union, overload

from ..api.chat.interfaces import ChatMessage
from ..controller.controller import ConditionalGroup, Controller
from ..exceptions import InferenceAbortedException
from ..features.features import Feature, FeatureEdits, FeatureGroup

SUPPORTED_MODELS = Literal[
    "meta-llama/Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"
]

ACTIVATION_NORMALIZATION_FACTOR = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 9,
    "meta-llama/Llama-3.3-70B-Instruct": 12,
}

ScopeCallable = Callable[["NestedScope"], "NestedScope"]
HandlerCallable = Callable[["InferenceContext"], "InferenceContext"]


class Variant:
    """A class representing a variant of a base model with feature modifications.

    This class allows for creating variants of a base model by applying
    feature modifications through either nudging or pinning values.

    Args:
        base_model (str): Identifier of the base model to create variants from

    Attributes:
        base_model (str): The base model identifier
        edits (FeatureEdits): Collection of feature modifications
    """

    def __init__(self, base_model: SUPPORTED_MODELS):
        self.base_model = base_model
        self.edits: FeatureEdits = FeatureEdits({})
        self.scopes: list[NestedScope] = []
        self._handlers: dict[str, HandlerCallable] = {}

    @overload
    def set(self, feature: FeatureEdits, value: float):
        """Set or modify feature values in the variant.

        Args:
            feature (FeatureEdits): Feature(s) to modify
            value (float): Value to apply
        """
        ...

    @overload
    def set(self, feature: dict[Feature, float]):
        """Set or modify feature values in the variant.

        Args:
            feature (Union[Feature, FeatureGroup]): Feature(s) to modify
            value (Union[float, None]): Value to apply:
                - float: For numerical adjustments
                - None: To clear the modification
        """
        ...

    @overload
    def set(self, feature: Union[Feature, FeatureGroup], value: float):
        """Set or modify feature values in the variant.

        Args:
            feature (Union[Feature, FeatureGroup]): Feature(s) to modify
            value (Union[float, None]): Value to apply:
                - float: For numerical adjustments
                - None: To clear the modification
        """
        ...

    def set(
        self,
        *args,
    ):
        if len(args) == 1:
            edits = args[0]
            if isinstance(args[0], FeatureEdits):
                edits = args[0].as_dict()

            for feature, value in edits.items():
                self.set(feature, value)

            return

        feature, value = args
        if value is None:
            self.clear(feature)
            return

        if isinstance(feature, Feature):
            self.edits.set(feature, value)
        elif isinstance(feature, FeatureGroup):
            for f in feature:
                self.set(f, value)
        else:
            raise NotImplementedError(f"Invalid feature type: {type(feature)}")

    def clear(self, feature: Union[Feature, FeatureGroup]):
        """Remove modifications for specified feature(s).

        Args:
            feature (Union[Feature, FeatureGroup]): Feature(s) to clear modifications for
        """
        if isinstance(feature, Feature):
            self.edits.remove(feature)
        else:
            for f in feature:
                self.edits.remove(f)

    def reset(self):
        """Remove all feature modifications."""
        self.edits.reset()
        self.scopes = []

    def __repr__(self):
        return str(self)

    def __str__(self):
        edits = "{"
        for feature, edit in self.edits._edits.items():
            edits += f"\n      {feature}: {edit},"
        edits += "\n   }"

        scopes = "{"
        for scope in self.scopes:
            scope_str = str(scope).replace("\n", "\n      ")
            scopes += f"\n      {scope_str},"
        scopes += "\n   }"

        return f"Variant(\n   base_model={self.base_model},\n   edits={edits}\n   scopes={scopes}\n)"

    @classmethod
    def from_json(cls, variant_json: Union[str, dict]):
        if isinstance(variant_json, str):
            variant_json = json.loads(variant_json)

        variant = Variant(variant_json["base_model"])
        for edit in variant_json["edits"]:
            feature = Feature(
                uuid=edit["feature_id"],
                label=edit["feature_label"],
                index_in_sae=edit["index_in_sae"],
            )
            variant.set(feature, edit["value"])

        for scope in variant_json["scopes"]:
            variant.scopes.append(NestedScope.from_json(scope))

        return variant

    def json(self):
        """Convert the variant to a JSON-compatible dictionary.

        Returns:
            dict: Dictionary containing base model and feature configurations
        """
        return {
            "base_model": self.base_model,
            "edits": [
                {
                    "feature_id": str(feature.uuid),
                    "feature_label": feature.label,
                    "index_in_sae": feature.index_in_sae,
                    "value": edit,
                }
                for feature, edit in self.edits._edits.items()
            ],
            "scopes": [scope.json() for scope in self.scopes],
        }

    def set_when(
        self,
        condition: ConditionalGroup,
        values: Union[FeatureEdits, dict[Union[Feature, FeatureGroup], float]],
    ) -> None:
        scope = NestedScope(condition, self)

        if isinstance(values, FeatureEdits):
            values = values.as_dict()

        for feature, value in values.items():
            scope.set(feature, value)

        self.scopes.append(scope)

    def abort_when(self, condition: ConditionalGroup) -> None:
        def _abort_handler(context: InferenceContext) -> None:
            raise InferenceAbortedException(
                f"Aborted inference due to conditional check:\n {condition}"
            )

        self.handle_when(condition, _abort_handler)

    def handle_when(
        self, condition: ConditionalGroup, handler: HandlerCallable
    ) -> None:
        event_name = str(uuid.uuid4())
        self._handlers[event_name] = handler
        scope = NestedScope(condition, self, event_name=event_name)
        self.scopes.append(scope)

    @property
    def controller(self) -> Controller:
        """Get a controller instance with the variant's modifications applied.

        Returns:
            Controller: Controller instance with feature modifications
        """
        controller = Controller()

        for feature, value in self.edits.as_dict().items():
            controller[feature] += value
        for scope in self.scopes:
            with controller.when(scope.condition) as ctl_scope:
                if scope.event_name:
                    ctl_scope.interrupt(scope.event_name)
                    continue

                for feature, value in scope._nested_variant.edits.as_dict().items():
                    controller[feature] += value

        return controller


class NestedScope:
    def __init__(
        self,
        condition: ConditionalGroup,
        base_variant: Variant,
        event_name: Optional[str] = None,
    ):
        self.event_name: Optional[str] = event_name
        self.condition = condition
        self._nested_variant = Variant(base_variant.base_model)

        self.set = self._nested_variant.set
        self.clear = self._nested_variant.clear
        self.reset = self._nested_variant.reset

    def json(self):
        return {
            "condition": self.condition,
            "nested_variant": self._nested_variant.json(),
        }

    @classmethod
    def from_json(cls, nested_scope_json: Union[str, dict]):
        if isinstance(nested_scope_json, str):
            nested_scope_json = json.loads(nested_scope_json)

        scope = NestedScope(
            condition=ConditionalGroup.from_json(nested_scope_json["condition"]),
            base_variant=Variant(nested_scope_json["base_model"]),
        )
        scope._nested_variant = Variant.from_json(nested_scope_json["nested_variant"])
        return scope

    def __str__(self):
        formatted_condition = str(self.condition).replace("\n", "\n   ")
        formatted_nested_variant = str(self._nested_variant).replace("\n", "\n   ")
        return f"NestedScope(\n   condition={formatted_condition},\n   nested_variant={formatted_nested_variant},\n   event_name={self.event_name}\n)"


@dataclass
class InferenceContext:
    prompt: list[ChatMessage]
    response_so_far: str

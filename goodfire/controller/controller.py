from dataclasses import dataclass
from typing import Any, Optional, Type, Union
from uuid import uuid4

from ..features.features import Conditional, ConditionalGroup, Feature, FeatureGroup
from .interfaces import INTERVENTION_MODE


@dataclass
class Intervention:
    mode: INTERVENTION_MODE
    features: FeatureGroup
    value: float

    def json(self, scale: float = 1) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "features": self.features.json(),
            "value": (
                self.value.json()
                if not isinstance(self.value, float)
                else self.value * scale
            ),
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Intervention":
        return Intervention(
            mode=data["mode"],
            features=FeatureGroup.from_json(data["features"]),
            value=data["value"],
        )

    def _prepare_values_for_stringification(self):
        if len(self.features) == 1:
            variable = str(self.features._features[0])
        else:
            variable = str(self.features).replace("\n", "\n   ")

        if self.mode == "nudge":
            operator = "+=" if self.value > 0 else "-"
            value = abs(self.value)
        elif self.mode == "pin":
            operator = "="
            value = self.value
        elif self.mode == "mul":
            operator = "*="
            value = self.value
        else:
            raise NotImplementedError()

        return variable, operator, value

    def __repr__(self):
        return str(self)

    def __str__(self):
        variable, operator, value = self._prepare_values_for_stringification()
        return f'Intervention(\n   feature={variable},\n   operator="{operator}",\n   value={value}\n)'

    def as_code(self):
        variable, operator, value = self._prepare_values_for_stringification()
        return f"{variable} {operator} {value}"


class InterventionBuffer:
    def __init__(self, controller: "Controller"):
        self._controller = controller

    def __str__(self):
        output_string = "InterventionBuffer(["
        for index, intervention in enumerate(self._controller._interventions[:9]):
            formatted_intervetion = intervention.as_code().replace("\n", "\n   ")
            output_string += f"\n   {index}: {formatted_intervetion};"

        if len(self._controller._interventions) > 9:
            output_string += "\n   ..."
            formatted_intervetion = (
                self._controller._interventions[-1].as_code().replace("\n", "\n   ")
            )
            output_string += f"\n   {len(self._controller._interventions) - 1}: {formatted_intervetion},"

        output_string += "\n])"
        return output_string

    def __repr__(self):
        return str(self)

    def pop(self, index: int):
        intervention = self._controller._interventions.pop(index)

        return intervention

    def __getitem__(self, index: int):
        return self._controller._interventions[index]

    def __len__(self):
        return len(self._controller._interventions)

    def __iter__(self):
        return iter(self._controller._interventions)

    def __contains__(self, intervention: Intervention):
        return intervention in self._controller._interventions

    def push(self, intervention: Intervention):
        self._controller._interventions.append(intervention)

    def insert(self, intervention: Intervention, index: int = -1):
        self._controller._interventions.insert(index, intervention)

    def empty(self):
        self._controller._interventions = []


@dataclass
class Scope:
    conditionals: "ConditionalGroup"
    controller: "ScopedController"
    is_active: bool = False
    event_name: Optional[str] = None

    def json(self, scale: float = 1) -> dict[str, Any]:
        return {
            "conditionals": self.conditionals.json(scale=scale),
            "controller": self.controller.json(scale=scale),
            "is_active": self.is_active,
            "event_name": self.event_name,
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Scope":
        return Scope(
            conditionals=ConditionalGroup.from_json(data["conditionals"]),
            controller=ScopedController.from_json(data["controller"]),
            is_active=data["is_active"],
            event_name=data["event_name"],
        )

    def interrupt(self, event_name: str):
        self.event_name = event_name


class Controller:
    def __init__(self, _parent_controller: Optional["Controller"] = None):
        self._interventions: list[Intervention] = []
        self._scopes: list[Scope] = []
        self._nonzero_strength_threshold: Optional[float] = None
        self._min_nudge_entropy: Optional[float] = None
        self._max_nudge_entropy: Optional[float] = None
        self._active_scope: Optional[Scope] = None
        self._parent_controller: Optional[Controller] = _parent_controller

        self.name = "controller__" + str(id(self))[:8]
        self.id = str(uuid4())

    @property
    def buffer(self):
        return InterventionBuffer(self)

    def json(self, scale: float = 1) -> dict[str, Any]:
        return {
            "interventions": [
                intervention.json(scale=scale) for intervention in self._interventions
            ],
            "scopes": [scope.json(scale=scale) for scope in self._scopes],
            "name": self.name,
            "nonzero_strength_threshold": self._nonzero_strength_threshold,
            "min_nudge_entropy": self._min_nudge_entropy,
            "max_nudge_entropy": self._max_nudge_entropy,
        }

    def __str__(self):
        return f'Controller(name="{self.name}", id="{self.id}")'

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_json(
        data: dict[str, Any],
        name: Optional[str] = None,
        id: Optional[str] = None,
        _controller_cls: Optional[Type["Controller"]] = None,
    ) -> "Controller":
        controller = _controller_cls() if _controller_cls else Controller()
        if not name:
            name = f"controller__{str(uuid4())[:8]}" if not name else name
        controller.name = name
        if not id:
            id = f"controller__{str(uuid4())[:8]}" if not id else id
        controller.id = id

        if data:
            controller._interventions = [
                Intervention.from_json(intervention_data)
                for intervention_data in data.get("interventions", [])
            ]
            controller._scopes = [
                Scope.from_json(scope_data) for scope_data in data.get("scopes", [])
            ]

            controller._nonzero_strength_threshold = data.get(
                "nonzero_strength_threshold", None
            )
            controller._min_nudge_entropy = data.get("min_nudge_entropy", None)
            controller._max_nudge_entropy = data.get("max_nudge_entropy", None)

        return controller

    def when(self, conditional: Union["Conditional", "ConditionalGroup"]):
        initial_parent_controller = (
            self._active_scope.controller if self._active_scope else self
        )
        initial_scope = self._active_scope

        if isinstance(conditional, Conditional):
            conditional_group = ConditionalGroup([conditional])
        else:
            conditional_group: ConditionalGroup = conditional

        class InterventionScope:
            def __enter__(self):
                scope = Scope(
                    conditionals=conditional_group,
                    controller=ScopedController(
                        _parent_controller=initial_parent_controller
                    ),
                )
                initial_parent_controller._active_scope = scope

                parent_controller = initial_parent_controller
                while parent_controller._parent_controller:
                    parent_controller = parent_controller._parent_controller
                    parent_controller._active_scope = scope

                return scope

            def __exit__(self, exc_type, exc_val, exc_tb):
                if not exc_type:
                    if initial_parent_controller._active_scope:
                        initial_parent_controller._scopes.append(
                            initial_parent_controller._active_scope
                        )

                    initial_parent_controller._active_scope = initial_scope

                    parent_controller = initial_parent_controller
                    while parent_controller._parent_controller:
                        parent_controller = parent_controller._parent_controller
                        parent_controller._active_scope = initial_scope

        return InterventionScope()

    def __setitem__(
        self,
        key: Union[Feature, FeatureGroup],
        value: Union[float, int, bool, "InterventionProxy"],
    ):
        if isinstance(value, InterventionProxy):
            return

        if isinstance(value, bool):
            value = 0.5 if value else -0.5

        if isinstance(value, int):
            value = float(value)

        if isinstance(key, Feature):
            key = FeatureGroup([key])

        self._add_intervention(key, value, "pin")

    def __getitem__(self, key: Union[Feature, FeatureGroup]) -> "InterventionProxy":
        if isinstance(key, Feature):
            key = FeatureGroup([key])
        return InterventionProxy(self, key)

    def _add_intervention(
        self,
        features: FeatureGroup,
        value: float,
        mode: INTERVENTION_MODE = "nudge",
    ):
        interventions = self._interventions

        if self._active_scope:
            interventions = self._active_scope.controller._interventions

        if isinstance(value, int):
            value = float(value)

        for feature in features:
            interventions.append(
                Intervention(mode=mode, features=FeatureGroup([feature]), value=value)
            )

    def __eq__(self, other):
        if isinstance(other, Controller):
            if self._interventions == other._interventions:
                return True
        return False


class InterventionProxy:
    def __init__(self, controller: Controller, features: FeatureGroup):
        self.controller = controller
        self.features = features

    def __iadd__(self, value: float) -> "InterventionProxy":
        self.controller._add_intervention(self.features, value, mode="nudge")
        return self

    def __isub__(self, value: float) -> "InterventionProxy":
        self.controller._add_intervention(self.features, -value, mode="nudge")
        return self

    # Mul and div are not used right now in the activation function.
    def __imul__(self, value: float) -> "InterventionProxy":
        self.controller._add_intervention(self.features, value, mode="mul")
        return self

    def __truediv__(self, value: float) -> "InterventionProxy":
        self.controller._add_intervention(self.features, value**-1, mode="mul")
        return self


class ScopedController(Controller):
    @staticmethod
    def from_json(data: dict[str, Any]) -> "ScopedController":
        controller = ScopedController()
        controller._interventions = [
            Intervention.from_json(intervention_data)
            for intervention_data in data["interventions"]
        ]
        controller._scopes = [
            Scope.from_json(scope_data) for scope_data in data["scopes"]
        ]
        return controller

from collections import OrderedDict
from typing import Any, Optional, Union, overload
from uuid import UUID

from .interfaces import CONDITIONAL_OPERATOR, JOIN_OPERATOR


class FeatureNotInGroupError(Exception):
    pass


class Feature:
    """A class representing a single feature aka a conceptual unit of the SAE.

    Handles individual feature operations and comparisons. Features can be combined
    into groups and compared using standard operators.

    Attributes:
        uuid (UUID): Unique identifier for the feature
        label (str): Human-readable label describing the feature
        training dataset
        index_in_sae (int): Index position in the SAE
    """

    def __init__(self, uuid: UUID, label: str, index_in_sae: int):
        """Initialize a new Feature instance.

        Args:
            uuid: Unique identifier for the feature
            label: Human-readable label describing the feature
            index_in_sae: Index position in the SAE
        """
        self.uuid = uuid
        self.label = label
        self.index_in_sae = index_in_sae

    def json(self):
        return {
            # Change to hex while passing through http.
            "uuid": self.uuid.hex if isinstance(self.uuid, UUID) else self.uuid,
            "label": self.label,
            "index_in_sae": self.index_in_sae,
            "max_activation_strength": 1,
        }

    @staticmethod
    def from_json(data: dict[str, Any]):
        # If str is provided, update it to UUID.
        if isinstance(data["uuid"], str):
            data["uuid"] = UUID(data["uuid"])
        return Feature(
            uuid=data["uuid"],
            label=data["label"],
            index_in_sae=data["index_in_sae"],
        )

    def __or__(self, other: "Feature"):
        group = FeatureGroup()
        group.add(self)
        group.add(other)

        return group

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(self.uuid)

    def __str__(self):
        return f'Feature("{self.label}")'

    def __eq__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        return FeatureGroup([self]) == other

    def __ne__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        return FeatureGroup([self]) != other

    def __le__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        return FeatureGroup([self]) <= other

    def __lt__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        return FeatureGroup([self]) < other

    def __ge__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        return FeatureGroup([self]) >= other

    def __gt__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        return FeatureGroup([self]) > other


class FeatureGroup:
    """A collection of Feature instances with group operations.

    Provides functionality for managing and operating on groups of features, including
    union and intersection operations, indexing, and comparison operations.

    Example:
        >>> feature_group = FeatureGroup([feature1, feature2, feature3, feature4])
        >>> # Access single feature by index
        >>> first_feature = feature_group[0]  # Returns Feature
        >>>
        >>> # Slice features
        >>> first_two = feature_group[0:2]  # Returns FeatureGroup with features 0,1
        >>> last_two = feature_group[-2:]   # Returns FeatureGroup with last 2 features
        >>>
        >>> # Multiple indexes using list or tuple
        >>> selected = feature_group[[0, 2]]  # Returns FeatureGroup with features 0,2
        >>> selected = feature_group[0, 3]    # Returns FeatureGroup with features 0,3
    """

    def __init__(self, features: Optional[list["Feature"]] = None):
        self._features: OrderedDict[int, "Feature"] = OrderedDict()

        if features:
            for feature in features:
                self.add(feature)

    def __iter__(self):
        for feature in self._features.values():
            yield feature

    def __hash__(self):
        return hash(frozenset(self._features.values()))

    @overload
    def __getitem__(self, index: int) -> "Feature": ...

    @overload
    def __getitem__(self, index: list[int]) -> "FeatureGroup": ...

    @overload
    def __getitem__(self, index: slice) -> "FeatureGroup": ...

    @overload
    def __getitem__(self, index: tuple[int, ...]) -> "FeatureGroup": ...

    def __getitem__(self, index: Union[int, list[int], tuple[int, ...], slice]):
        if isinstance(index, int):
            if index not in self._features:
                raise FeatureNotInGroupError(f"Feature with ID {index} not in group.")
            return self._features[index]
        elif isinstance(index, list) or isinstance(index, tuple):
            if isinstance(index, tuple):
                index = list(index)
            features: list[Feature] = []
            failed_indexes: list[int] = []
            while len(index) > 0:
                latest_index = index.pop(0)
                try:
                    features.append(self._features[latest_index])
                except KeyError:
                    failed_indexes.append(latest_index)

            if len(failed_indexes) > 0:
                raise FeatureNotInGroupError(
                    f"Features with IDs {failed_indexes} not in group."
                )

            return FeatureGroup(features)
        else:
            start = index.start if index.start else 0
            stop = index.stop if index.stop else len(self._features)
            step = index.step if index.step else 1

            if start < 0:
                start = len(self._features) + start

            if stop < 0:
                stop = len(self._features) + stop

            if step < 0:
                start, stop = stop, start

            if stop > len(self._features):
                stop = len(self._features)

            if start > len(self._features):
                start = len(self._features)

            if step == 0:
                raise ValueError("Step cannot be zero.")

            return FeatureGroup([self._features[i] for i in range(start, stop, step)])

    def __repr__(self):
        return str(self)

    def pick(self, feature_indexes: list[int]):
        """Create a new FeatureGroup with selected features.

        Args:
            feature_indexes: List of indexes to select

        Returns:
            FeatureGroup: New group containing only the selected features
        """
        new_group = FeatureGroup()
        for index in feature_indexes:
            new_group.add(self._features[index])

        return new_group

    def json(self) -> dict[str, Any]:
        return {"features": [f.json() for f in self._features.values()]}

    @staticmethod
    def from_json(data: dict[str, Any]):
        if not data.get("features"):
            return FeatureGroup([Feature.from_json(data)])
        return FeatureGroup([Feature.from_json(f) for f in data["features"]])

    def add(self, feature: "Feature"):
        """Add a feature to the group.

        Args:
            feature: Feature instance to add to the group
        """
        keys = list(self._features.keys())
        key_index = keys[-1] + 1 if keys else 0
        self._features[key_index] = feature

    def pop(self, index: int):
        """Remove and return a feature at the specified index.

        Args:
            index: Index of the feature to remove

        Returns:
            Feature: The removed feature
        """
        feature = self._features[index]
        del self._features[index]

        return feature

    def union(self, feature_group: "FeatureGroup"):
        """Combine this group with another feature group.

        Args:
            feature_group: Another FeatureGroup to combine with

        Returns:
            FeatureGroup: New group containing features from both groups
        """
        new_group = FeatureGroup()

        new_features: OrderedDict[int, Feature] = OrderedDict()

        for index, feature in self._features.items():
            new_features[index] = feature

        for index, feature in feature_group._features.items():
            new_features[len(self._features) + index] = feature

        new_group._features = new_features

        return new_group

    def intersection(self, feature_group: "FeatureGroup"):
        """Create a new group with features common to both groups.

        Args:
            feature_group: Another FeatureGroup to intersect with

        Returns:
            FeatureGroup: New group containing only features present in both groups
        """
        new_group = FeatureGroup()
        new_features: OrderedDict[int, Feature] = OrderedDict()

        index_in_new_group = 0
        other_group_uuids = {feat.uuid for feat in feature_group._features.values()}
        for _, feature in self._features.items():
            if feature.uuid in other_group_uuids:
                new_features[index_in_new_group] = feature
                index_in_new_group += 1

        new_group._features = new_features

        return new_group

    def __or__(self, other: "FeatureGroup"):
        return self.union(other)

    def __and__(self, other: "FeatureGroup"):
        return self.intersection(other)

    def __len__(self):
        return len(self._features)

    def __str__(self):
        features = list(self._features.items())
        if len(features) <= 10:
            features_str = ",\n   ".join(
                [f'{index}: "{f.label}"' for index, f in features[:10]]
            )
        else:
            features_str = ",\n   ".join(
                [f'{index}: "{f.label}"' for index, f in features[:9]]
            )
            features_str += ",\n   ...\n   "
            features_str += ",\n   ".join(
                [f'{index}: "{f.label}"' for index, f in features[-1:]]
            )

        return f"FeatureGroup([\n   {features_str}\n])"

    def __eq__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        if isinstance(other, Feature):
            return self == FeatureGroup([other])
        else:
            return Conditional(self, other, "==")

    def __ne__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        if isinstance(other, Feature):
            return self != FeatureGroup([other])
        else:
            return Conditional(self, other, "!=")

    def __le__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        if isinstance(other, Feature):
            return self <= FeatureGroup([other])
        else:
            return Conditional(self, other, "<=")

    def __lt__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        if isinstance(other, Feature):
            return self < FeatureGroup([other])
        else:
            return Conditional(self, other, "<")

    def __ge__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        if isinstance(other, Feature):
            return self >= FeatureGroup([other])
        else:
            return Conditional(self, other, ">=")

    def __gt__(
        self,
        other: Union[
            "FeatureGroup",
            "Feature",
            float,
        ],
    ) -> "Conditional":
        if isinstance(other, Feature):
            return self > FeatureGroup([other])
        else:
            return Conditional(self, other, ">")


class ConditionalGroup:
    """Groups multiple conditions with logical operators.

    Manages groups of conditions that can be combined using AND/OR operations.
    """

    def __init__(
        self, conditionals: list["Conditional"], operator: JOIN_OPERATOR = "AND"
    ):
        """Initialize a new ConditionalGroup.

        Args:
            conditionals: List of Conditional instances to group
            operator: Logical operator to join conditions ("AND" or "OR")
        """
        self.conditionals = conditionals
        self.operator = operator

    def json(self, scale: float = 1) -> dict[str, Any]:
        """Convert the conditional group to a JSON-serializable dictionary.

        Returns:
            dict: Dictionary containing conditionals and operator
        """
        return {
            "conditionals": [c.json(scale=scale) for c in self.conditionals],
            "operator": self.operator,
        }

    @staticmethod
    def from_json(data: dict[str, Any]):
        """Create a ConditionalGroup instance from JSON data.

        Args:
            data: Dictionary containing conditionals and operator

        Returns:
            ConditionalGroup: New instance with the deserialized data
        """
        return ConditionalGroup(
            [Conditional.from_json(c) for c in data["conditionals"]],
            operator=data["operator"],
        )

    def __and__(
        self, other: Union["ConditionalGroup", "Conditional"]
    ) -> "ConditionalGroup":
        if isinstance(other, Conditional):
            other_group = ConditionalGroup([other])
        else:
            other_group: ConditionalGroup = other

        return ConditionalGroup(
            self.conditionals + other_group.conditionals, operator="AND"
        )

    def __or__(
        self, other: Union["ConditionalGroup", "Conditional"]
    ) -> "ConditionalGroup":
        if isinstance(other, Conditional):
            other_group = ConditionalGroup([other])
        else:
            other_group: ConditionalGroup = other

        return ConditionalGroup(
            self.conditionals + other_group.conditionals, operator="OR"
        )

    def __str__(self):
        output = "ConditionalGroup([\n"
        for index, conditional in enumerate(self.conditionals):
            show_operator = "" if index == 0 else f"{self.operator.lower()} "
            conditional_str = str(conditional).replace("\n", "\n    ")
            output += f"    {show_operator}{conditional_str}"
            output += "\n"
        output += "])"
        return output

    def __repr__(self):
        return str(self)

    def __getitem__(self, index: Union[int, slice, tuple[int, ...]]):
        if isinstance(index, int):
            return self.conditionals[index]
        else:
            return ConditionalGroup(self.conditionals[index], operator=self.operator)

    def __len__(self):
        return len(self.conditionals)

    def __contains__(self, item: "Conditional"):
        return item in self.conditionals

    def __iter__(self):
        return iter(self.conditionals)

    def __mul__(self, other: float):
        return ConditionalGroup(
            [c * other for c in self.conditionals], operator=self.operator
        )

    def __rmul__(self, other: float):
        return self * other

    def __truediv__(self, other: float):
        return ConditionalGroup(
            [c / other for c in self.conditionals], operator=self.operator
        )

    def __rtruediv__(self, other: float):
        return self / other

    def __add__(self, other: float):
        return ConditionalGroup(
            [c + other for c in self.conditionals], operator=self.operator
        )

    def __radd__(self, other: float):
        return self + other

    def __sub__(self, other: float):
        return ConditionalGroup(
            [c - other for c in self.conditionals], operator=self.operator
        )

    def __rsub__(self, other: float):
        return self - other

    def __neg__(self):
        return ConditionalGroup([-c for c in self.conditionals], operator=self.operator)

    def __abs__(self):
        return ConditionalGroup(
            [abs(c) for c in self.conditionals], operator=self.operator
        )

    def __pow__(self, other: float):
        return ConditionalGroup(
            [c**other for c in self.conditionals], operator=self.operator
        )

    def __rpow__(self, other: float):
        return self**other


class Conditional:
    """Represents a conditional expression comparing features.

    Handles comparison operations between features, feature groups, and statistics.
    """

    def __init__(
        self,
        left_hand: FeatureGroup,
        right_hand: Union[Feature, FeatureGroup, float],
        operator: CONDITIONAL_OPERATOR,
    ):
        """Initialize a new Conditional.

        Args:
            left_hand: FeatureGroup for the left side of the comparison
            right_hand: Value to compare against (Feature, FeatureGroup, or float)
            operator: Comparison operator to use
        """

        self.left_hand = left_hand
        self.right_hand = right_hand
        self.operator = operator

    def json(self, scale: float = 1) -> dict[str, Any]:
        """Convert the conditional to a JSON-serializable dictionary.

        Returns:
            dict: Dictionary containing the conditional expression data
        """

        right_hand = self.right_hand

        if isinstance(right_hand, int) or isinstance(right_hand, float):
            right_hand = right_hand * scale

        return {
            "left_hand": self.left_hand.json(),
            "right_hand": (
                right_hand.json() if getattr(right_hand, "json", None) else right_hand
            ),
            "operator": self.operator,
        }

    @staticmethod
    def from_json(data: dict[str, Any]):
        """Create a Conditional instance from JSON data.

        Args:
            data: Dictionary containing conditional expression data

        Returns:
            Conditional: New instance with the deserialized data
        """
        return Conditional(
            FeatureGroup.from_json(data["left_hand"]),
            (
                Feature.from_json(data["right_hand"])
                if isinstance(data["right_hand"], dict)
                else data["right_hand"]
            ),
            data["operator"],
        )

    def __and__(self, other: "Conditional") -> ConditionalGroup:
        return ConditionalGroup([self, other], operator="AND")

    def __or__(self, other: "Conditional") -> ConditionalGroup:
        return ConditionalGroup([self, other], operator="OR")

    def __str__(self):
        left_hand_str = str(self.left_hand).replace("\n", "\n    ")
        right_hand_str = str(self.right_hand).replace("\n", "\n    ")
        return f"Conditional(\n   {left_hand_str} {self.operator} {right_hand_str}\n)"

    def __repr__(self):
        return str(self)

    def __mul__(self, other: float):
        return Conditional(self.left_hand, self.right_hand * other, self.operator)

    def __rmul__(self, other: float):
        return self * other

    def __truediv__(self, other: float):
        return Conditional(self.left_hand, self.right_hand / other, self.operator)

    def __rtruediv__(self, other: float):
        return self / other

    def __add__(self, other: float):
        return Conditional(self.left_hand, self.right_hand + other, self.operator)

    def __radd__(self, other: float):
        return self + other

    def __sub__(self, other: float):
        return Conditional(self.left_hand, self.right_hand - other, self.operator)

    def __rsub__(self, other: float):
        return self - other

    def __neg__(self):
        return Conditional(self.left_hand, -self.right_hand, self.operator)

    def __abs__(self):
        return Conditional(self.left_hand, abs(self.right_hand), self.operator)

    def __pow__(self, other: float):
        return Conditional(self.left_hand, self.right_hand**other, self.operator)

    def __rpow__(self, other: float):
        return self**other


class FeatureEdits:
    def __init__(self, edits: list[tuple[Feature, float]]):
        ordered_edits = OrderedDict()
        for key, value in edits:
            ordered_edits[key] = value

        self._edits: OrderedDict[Feature, float] = ordered_edits

    def __str__(self):
        output = "FeatureEdits(["
        for index, (feature, value) in enumerate(self._edits.items()):
            output += f"\n   {index}: ({feature.label}, {value})"
        return output + "\n])"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._edits)

    def __getitem__(self, index: Union[int, slice, tuple[int, ...]]):
        items = list(self._edits.items())
        if isinstance(index, int):
            return FeatureEdits([items[index]])
        elif isinstance(index, slice):
            return FeatureEdits(items[index])
        elif isinstance(index, tuple):
            return FeatureEdits([items[i] for i in index])
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    def __setitem__(self, index: int, value: tuple[Feature, float]):
        items = list(self._edits.items())
        items[index] = value
        new_dict = OrderedDict()
        for item in items:
            new_dict[item[0]] = item[1]
        self._edits = new_dict

    def set(self, feature: Feature, value: float):
        self._edits[feature] = value

    def remove(self, feature: Feature):
        del self._edits[feature]

    def rescale(self):
        def scale_weights(val: float, sum: float):
            return val / sum

        positive_weights = sum(value for value in self._edits.values() if value > 0)
        negative_weights = abs(
            sum(value for value in self._edits.values() if value < 0)
        )

        larger_weight = max(positive_weights, negative_weights)

        return FeatureEdits(
            [
                (feature, scale_weights(val, larger_weight) * 1.1)
                for feature, val in self._edits.items()
            ]
        )

    def reset(self):
        self._edits = OrderedDict()

    def __len__(self):
        return len(self._edits)

    def pop(self, index: int):
        return self._edits.pop(index)

    def __contains__(self, feature: Feature):
        return feature in self._edits

    def __eq__(self, other: "FeatureEdits"):
        return self._edits == other._edits

    def __ne__(self, other: "FeatureEdits"):
        return self._edits != other._edits

    def __hash__(self):
        return hash(frozenset(self._edits))

    def __mul__(self, other: float):
        return FeatureEdits(
            [(feature, value * other) for feature, value in self._edits.items()]
        )

    def __rmul__(self, other: float):
        return self * other

    def __truediv__(self, other: float):
        return FeatureEdits(
            [(feature, value / other) for feature, value in self._edits.items()]
        )

    def __rtruediv__(self, other: float):
        return self / other

    def __add__(self, other: "float"):
        return FeatureEdits(
            [(feature, value + other) for feature, value in self._edits.items()]
        )

    def __radd__(self, other: "float"):
        return self + other

    def __sub__(self, other: "float"):
        return FeatureEdits(
            [(feature, value - other) for feature, value in self._edits.items()]
        )

    def __rsub__(self, other: "float"):
        return self - other

    def __neg__(self):
        return FeatureEdits(
            [(feature, -value) for feature, value in self._edits.items()]
        )

    def __abs__(self):
        return FeatureEdits(
            [(feature, abs(value)) for feature, value in self._edits.items()]
        )

    def __pow__(self, other: float):
        return FeatureEdits(
            [(feature, value**other) for feature, value in self._edits.items()]
        )

    def __rpow__(self, other: float):
        return self**other

    def as_dict(self):
        return self._edits

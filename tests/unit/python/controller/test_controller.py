from random import randint
from uuid import uuid4

from goodfire.controller.controller import Controller
from goodfire.features.features import Feature, FeatureGroup


def _get_mock_feature():
    id = uuid4()
    return Feature(id, f"feature_{id}", randint(0, 100000))


def test_can_intervene_on_feature():
    controller = Controller()
    feature = _get_mock_feature()

    controller[feature] = 0.5

    assert controller._interventions[0].features == FeatureGroup([feature])
    assert controller._interventions[0].value == 0.5
    assert controller._interventions[0].mode == "pin"


def test_can_intervene_on_feature_group():
    controller = Controller()
    feature_group = _get_mock_feature() | _get_mock_feature()

    controller[feature_group] = 0.5

    assert controller._interventions[0].features == feature_group
    assert controller._interventions[0].value == 0.5


def test_can_conditionally_intervene__ge():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 >= 0.5):
        controller[feature_2] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5


def test_can_conditionally_intervene__le():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 <= 0.5):
        controller[feature_2] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == "<="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5


def test_can_conditionally_intervene__eq():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 == 0.5):
        controller[feature_2] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == "=="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5


def test_can_conditionally_intervene__ne():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 != 0.5):
        controller[feature_2] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == "!="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5


def test_can_conditionally_intervene__less_than():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 < 0.5):
        controller[feature_2] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == "<"

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5


def test_can_conditionally_intervene__greater_than():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 > 0.5):
        controller[feature_2] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">"

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5


def test_can_apply_interventions_before_and_after_scope():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()
    feature_3 = _get_mock_feature()

    controller[feature_1] = 0.5

    with controller.when(feature_1 >= 0.5):
        controller[feature_2] = 0.5

    controller[feature_3] = 0.5

    assert controller._interventions[0].features == FeatureGroup([feature_1])
    assert controller._interventions[0].value == 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5

    assert controller._interventions[1].features == FeatureGroup([feature_3])
    assert controller._interventions[1].value == 0.5


def test_can_apply_multiple_conditions():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()
    feature_3 = _get_mock_feature()

    with controller.when((feature_1 >= 0.5) & (feature_2 < 0.5)):
        controller[feature_3] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">="

    assert scope.conditionals.conditionals[1].left_hand == FeatureGroup([feature_2])
    assert scope.conditionals.conditionals[1].right_hand == 0.5
    assert scope.conditionals.conditionals[1].operator == "<"

    assert scope.controller._interventions[0].features == FeatureGroup([feature_3])
    assert scope.controller._interventions[0].value == 0.5


def test_can_apply_nested_conditionals():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 >= 0.5):
        with controller.when(feature_2 < 0.25):
            controller[feature_1] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">="

    nested_scope = scope.controller._scopes[0]
    assert nested_scope.conditionals.conditionals[0].left_hand == FeatureGroup(
        [feature_2]
    )
    assert nested_scope.conditionals.conditionals[0].right_hand == 0.25
    assert nested_scope.conditionals.conditionals[0].operator == "<"

    assert nested_scope.controller._interventions[0].features == FeatureGroup(
        [feature_1]
    )
    assert nested_scope.controller._interventions[0].value == 0.5


def test_can_exit_nested_conditional_and_still_apply_intervention():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    with controller.when(feature_1 >= 0.5):
        with controller.when(feature_2 < 0.25):
            controller[feature_2] = 0.25

        controller[feature_1] = 0.5

    scope = controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_1])
    assert scope.controller._interventions[0].value == 0.5

    nested_scope = scope.controller._scopes[0]
    assert nested_scope.conditionals.conditionals[0].left_hand == FeatureGroup(
        [feature_2]
    )
    assert nested_scope.conditionals.conditionals[0].right_hand == 0.25
    assert nested_scope.conditionals.conditionals[0].operator == "<"

    assert nested_scope.controller._interventions[0].features == FeatureGroup(
        [feature_2]
    )
    assert nested_scope.controller._interventions[0].value == 0.25


def test_can_do_addition_intervention():
    controller = Controller()
    feature_1 = _get_mock_feature()

    controller[feature_1] += 0.5

    assert controller._interventions[0].features == FeatureGroup([feature_1])
    assert controller._interventions[0].value == 0.5
    assert controller._interventions[0].mode == "nudge"


def test_can_do_subtraction_intervention():
    controller = Controller()
    feature_1 = _get_mock_feature()

    controller[feature_1] -= 0.5

    assert controller._interventions[0].features == FeatureGroup([feature_1])
    assert controller._interventions[0].value == -0.5
    assert controller._interventions[0].mode == "nudge"


def test_can_do_multiplication_intervention():
    controller = Controller()
    feature_1 = _get_mock_feature()

    controller[feature_1] *= 0.5

    assert controller._interventions[0].features == FeatureGroup([feature_1])
    assert controller._interventions[0].value == 0.5
    assert controller._interventions[0].mode == "mul"


def test_can_do_division_intervention():
    controller = Controller()
    feature_1 = _get_mock_feature()

    controller[feature_1] /= 0.5

    assert controller._interventions[0].features == FeatureGroup([feature_1])
    assert controller._interventions[0].value == 1 / 0.5
    assert controller._interventions[0].mode == "mul"


def test_can_set_to_another_feature():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()

    controller[feature_1] = feature_2

    assert controller._interventions[0].features == FeatureGroup([feature_1])
    assert controller._interventions[0].value == feature_2
    assert controller._interventions[0].mode == "pin"


def test_can_serialize_and_deserialize():
    controller = Controller()
    feature_1 = _get_mock_feature()
    feature_2 = _get_mock_feature()
    feature_3 = _get_mock_feature()

    controller[feature_3] = 0.75
    with controller.when(feature_1 >= 0.5):
        controller[feature_2] = 0.5

    data = controller.json()
    new_controller = Controller.from_json(data, name=data.get("name"))

    assert new_controller._interventions[0].features == FeatureGroup([feature_3])
    assert new_controller._interventions[0].value == 0.75

    scope = new_controller._scopes[0]
    assert scope.conditionals.conditionals[0].left_hand == FeatureGroup([feature_1])
    assert scope.conditionals.conditionals[0].right_hand == 0.5
    assert scope.conditionals.conditionals[0].operator == ">="

    assert scope.controller._interventions[0].features == FeatureGroup([feature_2])
    assert scope.controller._interventions[0].value == 0.5

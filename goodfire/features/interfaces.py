from typing import Literal

# Constants with type annotations
CONDITIONAL_OPERATOR = Literal[">", "<", "==", ">=", "<=", "!="]
"""Valid operators for feature comparisons: >, <, ==, >=, <=, !="""

JOIN_OPERATOR = Literal["AND", "OR"]
"""Logical operators for combining conditions: AND, OR"""

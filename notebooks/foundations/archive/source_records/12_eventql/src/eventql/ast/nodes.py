# nodes.py

from dataclasses import dataclass

class Node:
    pass

@dataclass
class Signal(Node):
    name: str

@dataclass
class Constant(Node):
    value: float

@dataclass
class Diff(Node):
    expr: Node

@dataclass
class Rise(Node):
    expr: Node

@dataclass
class Fall(Node):
    expr: Node

@dataclass
class Cumsum(Node):
    expr: Node


@dataclass
class Eq(Node):
    left: Node
    right: Node


@dataclass
class Lt(Node):
    left: Node
    right: Node


@dataclass
class Gt(Node):
    left: Node
    right: Node


@dataclass
class And(Node):
    left: Node
    right: Node


@dataclass
class Or(Node):
    left: Node
    right: Node

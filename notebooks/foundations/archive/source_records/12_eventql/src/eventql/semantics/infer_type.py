# infer_type.py

from eventql.semantics.types import DSLType
from eventql.ast.nodes import Signal, Constant, Diff, Cumsum, Eq, Lt, Gt, And, Or

def infer_type(node):

    if isinstance(node, Signal):
        return DSLType.SIGNAL

    if isinstance(node, Constant):
        return DSLType.SCALAR

    if isinstance(node, Diff):
        return DSLType.SIGNAL

    if isinstance(node, Cumsum):
        return DSLType.SIGNAL

    if isinstance(node, Eq):
        lt = infer_type(node.left)
        rt = infer_type(node.right)

        if lt == DSLType.SCALAR and rt == DSLType.SCALAR:
            return DSLType.SCALAR

        return DSLType.MASK

    if isinstance(node, Lt):
        lt = infer_type(node.left)
        rt = infer_type(node.right)

        if lt == DSLType.SCALAR and rt == DSLType.SCALAR:
            return DSLType.SCALAR

        return DSLType.MASK

    if isinstance(node, Gt):
        lt = infer_type(node.left)
        rt = infer_type(node.right)

        if lt == DSLType.SCALAR and rt == DSLType.SCALAR:
            return DSLType.SCALAR

        return DSLType.MASK

    if isinstance(node, And):
        return DSLType.MASK

    if isinstance(node, Or):
        return DSLType.MASK

    raise Exception(f"Unknown node: {node}")

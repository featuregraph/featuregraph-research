# interpreter.py

import torch
from eventql.ast.nodes import Signal, Constant, Diff, Cumsum, Eq, Lt, Gt, And, Or

def eval_value(node, ctx):

    if isinstance(node, Signal):
        return ctx[node.name]

    if isinstance(node, Constant):
        return torch.tensor(node.value)

    if isinstance(node, Diff):
        return torch.cat((torch.zeros(1), torch.diff(eval_value(node.expr, ctx))))

    if isinstance(node, Cumsum):
        return torch.cumsum(eval_value(node.expr, ctx), dim=0)

    if isinstance(node, Eq):
        return eval_value(node.left, ctx) == eval_value(node.right, ctx)

    if isinstance(node, Lt):
        return eval_value(node.left, ctx) < eval_value(node.right, ctx)

    if isinstance(node, Gt):
        return eval_value(node.left, ctx) > eval_value(node.right, ctx)

    if isinstance(node, And):
        return eval_value(node.left, ctx) & eval_value(node.right, ctx)

    if isinstance(node, Or):
        return eval_value(node.left, ctx) | eval_value(node.right, ctx)
# compiler.py

from eventql.compiler.expression import Expression
from eventql.runtime.interpreter import eval_value
from eventql.semantics.infer_type import infer_type

def compiler(node):

    t = infer_type(node)

    return Expression(
        node=node,
        type=t,
        evaluator=eval_value
    )
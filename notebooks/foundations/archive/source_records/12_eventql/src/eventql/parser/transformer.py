# transformer.py

from lark import Transformer
from eventql.ast.nodes import Signal, Constant, Diff, Cumsum, Eq, Lt, Gt, And, Or


class ASTBuilder(Transformer):

    def NAME(self, items):
        return Signal(str(items))

    def SIGNED_NUMBER(self, items):
        return Constant(float(items))

    def diff(self, items):
        return Diff(items[0])

    def cumsum(self, items):
        return Cumsum(items[0])

    def eq(self, items):
        return Eq(items[0], items[1])

    def lt(self, items):
        return Lt(items[0], items[1])

    def gt(self, items):
        return Gt(items[0], items[1])

    def and_expr(self, items):
        return And(items[0], items[1])

    def or_expr(self, items):
        return Or(items[0], items[1])

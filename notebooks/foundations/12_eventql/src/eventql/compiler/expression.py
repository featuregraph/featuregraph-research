# expression.py

from dataclasses import dataclass

@dataclass
class Expression:
    node: object
    type: object
    evaluator: callable

    def evaluate(self, ctx):
        return self.evaluator(self.node, ctx)

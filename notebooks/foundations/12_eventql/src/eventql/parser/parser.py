from importlib.resources import files
from lark import Lark

grammar = (
    files("eventql.parser")
    .joinpath("grammar.lark")
    .read_text()
)

parser = Lark(grammar, start="expr", parser="lalr")

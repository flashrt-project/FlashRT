"""Build-time bounded integer expressions for native execution uniforms.

The integer value is the maximum extent used for storage and launch capacity.
resolve() evaluates the current extent; only uniform contents change at replay.
"""

import operator

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "//": operator.floordiv}


def evaluate(expr, value):
    if expr == "value":
        return value
    if isinstance(expr, int):
        return expr
    op, left, right = expr
    return _OPS[op](evaluate(left, value), evaluate(right, value))


class Dimension(int):
    def __new__(cls, parameter, expression="value"):
        obj = super().__new__(cls, evaluate(expression, parameter.maximum))
        obj.parameter = parameter
        obj.expression = expression
        return obj

    def resolve(self):
        return evaluate(self.expression, self.parameter.value)

    def _binary(self, other, op, reverse=False):
        if isinstance(other, Dimension):
            if other.parameter is not self.parameter:
                raise ValueError("Shape expressions must share one bounded parameter")
            other = other.expression
        elif not isinstance(other, int):
            return NotImplemented
        left, right = (other, self.expression) if reverse else (self.expression, other)
        if op == "+" and left == 0:
            expr = right
        elif op in ("+", "-") and right == 0:
            expr = left
        elif op == "-" and left == right or op == "*" and (left == 0 or right == 0):
            return 0
        elif op == "*" and left == 1:
            expr = right
        elif op in ("*", "//") and right == 1:
            expr = left
        else:
            expr = (op, left, right)
        return Dimension(self.parameter, expr)

    def __add__(self, other):
        return self._binary(other, "+")

    def __radd__(self, other):
        return self._binary(other, "+", True)

    def __sub__(self, other):
        return self._binary(other, "-")

    def __rsub__(self, other):
        return self._binary(other, "-", True)

    def __mul__(self, other):
        return self._binary(other, "*")

    def __rmul__(self, other):
        return self._binary(other, "*", True)

    def __floordiv__(self, other):
        return self._binary(other, "//")

    def __rfloordiv__(self, other):
        return self._binary(other, "//", True)

    def __neg__(self):
        return self * -1


def resolve(value):
    return value.resolve() if isinstance(value, Dimension) else value


def uniform_key(value):
    if isinstance(value, Dimension):
        return (value.parameter, value.expression)
    return value

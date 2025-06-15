from enum import Enum
from typing import Protocol, Any, Iterator

class _SolverFactory:
    _cls: dict[str, Any]

    def __call__(self, solver_name: str) -> _Solver: ...

class _Solver(Protocol):
    def available(self) -> bool: ...
    def solve(self, model: ConcreteModel) -> _SolverResults: ...

class _Domain:
    pass

class _Var:
    def __init__(
        self,
        index_set: Any = None,
        *,
        domain: _Domain | None = None,
        bounds: tuple[float, float] | None = None,
        initialize: Any = None,
    ) -> None: ...
    def __getitem__(self, key: Any) -> _VarData: ...
    def __iter__(self) -> Iterator[_VarData]: ...
    def __rmul__(self, value: Any) -> _Var: ...
    def __mul__(self, value: Any) -> _Var: ...
    def __radd__(self, value: Any) -> _Var: ...
    def __add__(self, value: Any) -> _Var: ...
    def __rsub__(self, value: Any) -> _Var: ...
    def __sub__(self, value: Any) -> _Var: ...
    # Not sure if _Var is the correct return type for the comparison methods
    def __le__(self, value: Any) -> _Var: ...  # ???
    def __lt__(self, value: Any) -> _Var: ...  # ???
    def __ge__(self, value: Any) -> _Var: ...  # ???
    def __gt__(self, value: Any) -> _Var: ...  # ???
    def __eq__(self, value: Any) -> _Var: ...  # type: ignore[override]
    def __ne__(self, value: Any) -> _Var: ...  # type: ignore[override]

class _VarData:
    def __rmul__(self, value: Any) -> _Var: ...
    def __mul__(self, value: Any) -> _Var: ...
    def __radd__(self, value: Any) -> _Var: ...
    def __add__(self, value: Any) -> _Var: ...
    def __rsub__(self, value: Any) -> _Var: ...
    def __sub__(self, value: Any) -> _Var: ...
    # Not sure if _Var is the correct return type for the comparison methods
    def __le__(self, value: Any) -> _Var: ...  # ???
    def __lt__(self, value: Any) -> _Var: ...  # ???
    def __ge__(self, value: Any) -> _Var: ...  # ???
    def __gt__(self, value: Any) -> _Var: ...  # ???
    def __eq__(self, value: Any) -> _Var: ...  # type: ignore[override]
    def __ne__(self, value: Any) -> _Var: ...  # type: ignore[override]

class _Objective:
    def __init__(self, *, expr: Any, sense: "ObjectiveSense") -> None: ...

class ObjectiveSense(Enum):
    """Flag indicating if an objective is minimizing (1) or maximizing (-1).

    While the numeric values are arbitrary, there are parts of Pyomo
    that rely on this particular choice of value.  These values are also
    consistent with some solvers (notably Gurobi).

    """

    minimize = 1
    maximize = -1

maximize = ObjectiveSense.maximize
minimize = ObjectiveSense.minimize

class _Constraint:
    def __init__(self, *, expr: Any) -> None: ...

class ConcreteModel:
    def __init__(self) -> None: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class _SolverResults: ...

def assert_optimal_termination(results: _SolverResults) -> None: ...
def value(expr: Any) -> Any: ...

SolverFactory: _SolverFactory
NonNegativeReals: _Domain
Var = _Var
Objective = _Objective
Constraint = _Constraint

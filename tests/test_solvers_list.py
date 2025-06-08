from solvers_list import solvers_list


def test_solvers_list() -> None:
    assert "gdpopt" in solvers_list()

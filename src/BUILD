python_sources(
    dependencies = [
        "//stubs/pyomo:environ",
    ],
)

pex_binary(
    name = "main",
    dependencies = [
        "//:reqs#highspy",
        "//:reqs#pyomo",
    ],
    entry_point = "main:main",
    output_path = "food_powder_calculator.pex",
)

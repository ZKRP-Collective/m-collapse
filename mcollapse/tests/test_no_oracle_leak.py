from __future__ import annotations

import ast
import unittest
from pathlib import Path


class OracleBoundaryTests(unittest.TestCase):
    def test_relation_generator_has_no_oracle_parameter_name_or_import(self):
        source_path = Path(__file__).resolve().parents[1] / "relations.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        arguments = {
            argument.arg
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        imported_names = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }

        self.assertNotIn("oracle", names)
        self.assertNotIn("oracle", arguments)
        self.assertNotIn("oracle", imported_names)


if __name__ == "__main__":
    unittest.main()

import json
from typing import List, Tuple
from .models import FunctionDef, PromptTest


def load_inputs(functions_path: str,
                tests_path: str) -> Tuple[List[FunctionDef], List[PromptTest]]:
    """Loads function definitions and test prompts from JSON files."""
    try:
        with open(functions_path, 'r') as f:
            """transform json to dict"""
            raw_functions = json.load(f)
            """For each dictionary in raw_functions,
            create a FunctionDef object with its values."""
            functions = [FunctionDef(**fn) for fn in raw_functions]
        with open(tests_path, 'r') as f:
            raw_tests = json.load(f)
            tests = [PromptTest(**t) for t in raw_tests]

        return functions, tests

    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"Erreur lors du chargement des fichiers : {e}")
        return [], []

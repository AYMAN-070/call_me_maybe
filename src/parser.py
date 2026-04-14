import json
from typing import List, Tuple
from .models import FunctionDef, PromptTest
import sys


def read_json_safely(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error : File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error : file '{filepath}' contains malformed JSON : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error with '{filepath}': {e}")
        sys.exit(1)


def load_inputs(functions_path: str,
                tests_path: str) -> Tuple[List[FunctionDef], List[PromptTest]]:
    """Loads function definitions and test prompts from JSON files."""
    raw_functions = read_json_safely(functions_path)
    raw_tests = read_json_safely(tests_path)
    try:
        """For each dictionary in raw_functions,
        create a FunctionDef object with its values."""
        functions = [FunctionDef(**fn) for fn in raw_functions]
        tests = [PromptTest(**t) for t in raw_tests]
        return functions, tests

    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"Error while uploading file : {e}")
        sys.exit(1)

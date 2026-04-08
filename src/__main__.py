import argparse  # tool for handling args
import json
from llm_sdk import Small_LLM_Model


def main():
    parser = argparse.ArgumentParser(description="Function Calling")
    parser.add_argument("--functions_definition", default="data/input/"
                        "function_definitions.json")
    parser.add_argument("--input", default="data/input/"
                        "function_calling_tests.json")
    parser.add_argument("--output", default="data/output/"
                        "function_calling_results.json")
    args = parser.parse_args()
    try:
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"Success: Model loaded with {len(vocab)}"
              f"tokens in the vocabulary.")
        print(f"Ready for entry file : {args.input}")
    except Exception as e:
        print(f"Initialization error : {e}")


if __name__ == "__main__":
    main()

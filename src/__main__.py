import argparse  # tool for handling args
import json
from llm_sdk import Small_LLM_Model
from src.parser import load_inputs
from src.engine import generate_response


def main():
    parser = argparse.ArgumentParser(description="Function Calling")
    parser.add_argument("--functions_definition", default="data/input/"
                        "function_definitions.json")
    parser.add_argument("--input", default="data/input/"
                        "function_calling_tests.json")
    parser.add_argument("--output", default="data/output/"
                        "function_calling_results.json")
    # Read from terminal
    args = parser.parse_args()
    model = Small_LLM_Model()
    functions, prompts = load_inputs(args.functions_definition, args.input)
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    results = []
    for p in prompts:
        print(f"Processing : {p.prompt}")
        ai_json_str = generate_response(model, p.prompt, functions, vocab)
        print(f"\n--- TEXTE BRUT --- \n{ai_json_str}\n------------------\n")
        try:
            # AI text to dict
            ai_data = json.loads(ai_json_str)
            final_entry = {
                "prompt": p.prompt,
                "name": ai_data.get("name"),
                "parameters": ai_data.get("parameters")
            }
            results.append(final_entry)
        except Exception as e:
            print(f"Skipping due to AI format error: {e}")
    with open(args.output, 'w') as f:
        # write Python data to a JSON format in output file
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

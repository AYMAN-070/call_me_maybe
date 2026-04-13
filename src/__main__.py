import argparse  # tool for handling args
import json
from llm_sdk import Small_LLM_Model
from src.parser import load_inputs
from src.engine import generate_response


def main() -> None:
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
        print(f"\n--- TEXT GENERATED--- \n{ai_json_str}\n------------------\n")
        try:
            # AI text to dict
            ai_data = json.loads(ai_json_str)
            ai_func_name = ai_data.get("name")
            ai_params = ai_data.get("parameters", {})
            # Verify output Type
            target_function = next((f for f in functions
                                    if f.name == ai_func_name), None)
            if target_function:
                for arg_name, value in ai_params.items():
                    # Very if it's an int and IA make a string
                    if arg_name in target_function.parameters:
                        expected_type = target_function.parameters[arg_name].type
                        if expected_type == "number":
                            try:
                                ai_params[arg_name] = float(value)
                            except (ValueError, TypeError):
                                pass
            final_entry = {
                "prompt": p.prompt,
                "name": ai_func_name,
                "parameters": ai_params
            }
            results.append(final_entry)
        except Exception as e:
            print(f"Skipping due to AI format error: {e}")
    with open(args.output, 'w') as f:
        # write Python data to a JSON format in output filee
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

\*This project has been created as part of the 42 curriculum by ayhammou.*

# Call Me Maybe: Function Calling in LLMs

## Description
This project implements a function calling system that bridges the gap between natural language understanding and machine-executable structured outputs. Using the small-scale Qwen3-0.6B language model, this tool translates user prompts into precise, 100% valid JSON function calls containing the correct function name and typed arguments.

## Instructions

### Installation
To set up the project environment and install dependencies (numpy, pydantic), use the provided Makefile:
```bash
make install
# Alternatively, you can directly run: uv sync
```

### Example Usage
You can run the program using the default paths or specify custom ones. 
```bash
make run
```
Or manually via the CLI:
```bash
uv run python -m src \
  --functions_definition data/input/function_definitions.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Linting & Debugging
To ensure code quality, strict typing, and to run the project in debug mode:
```bash
make lint    # Runs flake8 and mypy strict checks
make debug   # Runs the script in debug mode
make clean   # Removes cache and temporary files
```

---

## Algorithm Explanation
This project uses **Constrained Decoding** to guarantee structural and semantic validity. Instead of relying on the LLM to spontaneously output correct JSON, the generation process is intercepted token-by-token:
1. **Logits Interception:** Before the model selects the next token, the probability scores (logits) of all possible tokens are analyzed.
2. **State Machine Tracking:** A custom `JSONState` class tracks the current position within the JSON structure (e.g., waiting for the function name, waiting for parameters).
3. **Prefix Matching Filtering:** When generating the function name, the algorithm compares the text generated so far against the vocabulary. Tokens that do not form a valid function name are penalized by setting their logits to negative infinity (`-np.inf`).
4. **Argmax Selection:** The model is forced to choose the most probable valid token remaining.

## Design Decisions
* **Numpy over Heavy Frameworks:** To comply with the strict requirement of not using PyTorch or external ML libraries, the logits modification is handled entirely using pure `numpy` arrays.
* **Pydantic Validation:** `pydantic` is used to dynamically validate the final JSON structure and ensure that argument types (like strings vs. numbers) strictly match the function definitions.
* **Prefix Matching over Exact Match:** To bypass the issue of subword tokenization (where a single word is split into multiple tokens by the model), the filtering algorithm checks for syllable prefixes rather than full exact words.
* **Prefilling:** The generation is explicitly primed with `{"name": "` to immediately lock the LLM into the desired JSON format, effectively optimizing the state machine and eliminating initial unpredictability.

## Performance Analysis
* **Reliability:** By hard-forcing the JSON structure and using state-based filtering, the system achieves 100% valid, parseable JSON outputs.
* **Accuracy:** The function selection process approaches near-perfect accuracy by restricting the token search space exclusively to the provided function schema.
* **Speed:** Processing is extremely fast, taking well under the 5-minute requirement, due to the lightweight nature of Numpy operations and the O(1) complexity optimizations in the state machine.

## Challenges Faced
* **The Tokenization Trap:** The most significant challenge was dealing with subword tokenization. The model rarely predicts a 15-character function name as a single token. This caused initial "infinite loop" bugs. *Solution:* Implementing a dynamic prefix-matching algorithm that evaluates token additions based on what has already been typed.
* **Edge Cases & NoneTypes:** Improper stop conditions initially caused `NoneType` return errors. *Solution:* Refactoring the main generation loop to ensure robust, fail-safe return statements even if the maximum token limit is reached.

## Testing Strategy
The implementation was validated using a multi-layered approach:
* **Linter Compliance:** Strict adherence to `flake8` and `mypy` standards to ensure type safety and prevent runtime crashes.
* **Provided Test Suites:** Extensive testing with the `function_calling_tests.json` to verify correct function selection and argument extraction.
* **Edge Case Injection:** Manual testing with numbers passed as strings and complex parameter requirements to ensure the `pydantic` models properly cast and validate the data.

## Resources
* Documentation on Subword Tokenization (BPE/SentencePiece).
* Python Official Documentation (`typing` module and Context Managers).
* Pydantic Documentation for dynamic model creation.
* **AI Usage:** Generative AI was used as a learning companion throughout the project. Specifically, it was used to:
    * Understand the mathematical relationship between Logits, Softmax, and Tokenization.
    * Refactor the State Machine to adhere to the KISS principle.
    * Troubleshoot and resolve strict `mypy` typing errors related to the provided SDK.
    * AI generated code was systematically peer-reviewed and rewritten to ensure deep personal comprehension before integration.

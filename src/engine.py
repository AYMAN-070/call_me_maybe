import numpy as np
from typing import List, Dict, Any
from .models import FunctionDef


class JSONState:
    """
    Manages the current state of the JSON structure to guide token filtering.
    """
    def __init__(self, allowed_functions: List[FunctionDef]) -> None:
        self.functions = [f.name for f in allowed_functions]
        self.current_step = "START"

    def update_state(self, last_text: str) -> None:
        """desactivate state filtre if fct name is generated"""
        if self.current_step != "FUNC_NAME":
            return
        value = last_text.split(":")[-1]
        # if name is done
        if value.count('"') >= 2:
            self.current_step = "PARAM_START"


def json_constraints(logits: np.ndarray, state: JSONState,
                     vocab: Dict[str, int], current_text: str) -> np.ndarray:
    """Filters logits based on the logical state
    Takes scores(logits) and kill forbiden words"""
    # take a copy of IA scores proba
    filtred_logits = np.copy(logits)
    if state.current_step != "FUNC_NAME":
        return filtred_logits
    filtred_logits.fill(-np.inf)
    try:
        typed_so_far = current_text.split('"name": "')[1]
    except IndexError:
        typed_so_far = ""
    allowed_targets = [name + '"' for name in state.functions]
    valid_token_found = False
    for token_str, token_id in vocab.items():
        # test new word with the syllabe
        candidate = typed_so_far + token_str
        for target in allowed_targets:
            if target.startswith(candidate) or candidate == target:
                filtred_logits[token_id] = logits[token_id]
                valid_token_found = True
                break
    # security
    if not valid_token_found:
        return logits
    return filtred_logits


def generate_response(model: Any, prompt: str, functions: List[FunctionDef],
                      vocab: Dict[str, int]) -> str:
    """Generates a complete JSON response using constraint decoding."""
    tools_doc = ""
    valid_names_fct = []
    for f in functions:
        tools_doc += f"- {f.name}: {f.description}. Parameters: {f.parameters}"
        valid_names_fct.append(f'"{f.name}"')
    allowed_names_str = ", ".join(valid_names_fct)
    system_instruction = (
        "System: You are a strict function calling AI.\n"
        f"CRITICAL: You MUST choose a tool name ONLY from this list: "
        f"{allowed_names_str}.\n"
        "Output ONLY a valid JSON object in this exact format:\n"
        '{"name": "the_tool_name", "parameters": {"arg_name": "value"}}\n\n'
        f"Available tools:\n{tools_doc}"
    )
    prefix = '{"name": "'
    full_prompt = f"{system_instruction}\nUser: {prompt}\nAssistant:\n{prefix}"
    # Encoding: We get a clean list of integers using [0].tolist()
    generated_ids: List[int] = model.encode(full_prompt)[0].tolist()
    state = JSONState(functions)
    state.current_step = "FUNC_NAME"
    final_gen_text: str = ""
    for i in range(200):
        full_text = model.decode(generated_ids)
        current_gen_text = full_text.split("Assistant:")[-1]
        prev_step = state.current_step
        state.update_state(current_gen_text)
        # INJECTION FORCÉE DE LA VIRGULE ET DES PARAMÈTRES
        if prev_step == "FUNC_NAME" and state.current_step == "PARAM_START":
            # On cherche l'endroit exact où l'IA a fermé le guillemet du nom
            start_idx = full_text.rfind('{"name": "')
            second_quote_idx = full_text.find('"', start_idx + 10)
            # On coupe les erreurs (espaces en trop, mots oubliés) et on FORCE la suite !
            clean_full_text = (full_text[:second_quote_idx + 1] +
                               ', "parameters": {')
            # On remet ce texte parfait dans la tête de l'IA et on relance !
            generated_ids = model.encode(clean_full_text)[0].tolist()
            continue
        #  get Logits
        raw_logits_list = model.get_logits_from_input_ids(generated_ids)
        # Transform the output into a NumPy array so you can easily manipulate
        # the data float32 is less memory than floqt64
        last_token_logits = np.array(raw_logits_list, dtype=np.float32)
        # apply my filtre
        filtered_logits = json_constraints(last_token_logits, state, vocab,
                                           current_gen_text)
        # chose the high proba
        next_token_id = int(np.argmax(filtered_logits))
        generated_ids.append(next_token_id)
        # Verify text with new token
        final_text = model.decode(generated_ids)
        final_gen_text = final_text.split("Assistant:\n")[-1].strip()
        # stop consition
        if ((next_token_id == model._tokenizer.eos_token_id) or
            (final_gen_text.count("{") == final_gen_text.count("}") and
             final_gen_text.count("{") > 0)):
            return final_gen_text
    return final_gen_text

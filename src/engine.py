import torch
from typing import List, Dict
from .models import FunctionDef


class JSONState:
    """
    Manages the current state of the JSON structure to guide token filtering.
    """
    def __init__(self, allowed_functions: List[FunctionDef]) -> None:
        self.functions = [f.name for f in allowed_functions]
        self.current_step = "START"

    def update_state(self, last_text: str):
        """next state based on the text already generated"""
        clean_text = last_text.strip()
        if clean_text.endswith("{"):
            self.current_step = "KEY_NAME"
        elif clean_text.endswith('"name"'):
            self.current_step = "COLON"
        elif clean_text.endswith(':'):
            self.current_step = "FUNC_NAME"
        elif self.current_step == "FUNC_NAME" and '"' in clean_text.split(':')[-1]:
            self.current_step = "COMMA"
        elif clean_text.endswith(",") and self.current_step == "COMMA":
            self.current_step = "KEY_ARGS"
        elif clean_text.endswith('"parameters"'):
            self.current_step = "ARG_COLON"
        elif clean_text.endswith(':') and self.current_step == "ARG_COLON":
            self.current_step = "ARG_START"
        elif clean_text.endswith('{') and self.current_step == "ARG_START":
            self.current_step = "VALUE"
        elif clean_text.endswith('}'):
            self.current_step = "END"

def json_constraints(logits: torch.Tensor, state: JSONState, vocab: Dict) -> torch.Tensor:
    """Filters logits based on the logical state
    Takes scores(logits) and kill forbiden words"""
    # take a copy of IA scores
    original_logits = logits.clone()
    if state.current_step != "FUNC_NAME":
        return original_logits
    logits.fill_(-float('inf'))
    for fn_name in state.functions:
        target = f'"{fn_name}"'
        token_id = vocab.get(target)
        if token_id is not None: 
            logits[token_id] = original_logits[token_id]
    if torch.all(logits == -float('inf')):
        return original_logits
    return logits

def generate_response(model, prompt: str, functions: List[FunctionDef], vocab: Dict):
    """Generates a complete JSON response using constraint decoding."""
    tools_doc = ""
    valid_names_fct = []
    for f in functions:
        tools_doc += f"- {f.name}: {f.description}. Parameters: {f.parameters}\n"
        valid_names_fct.append(f'"{f.name}"')
    allowed_names_str = ", ".join(valid_names_fct)
    system_instruction = (
        "System: You are a strict function calling AI.\n"
        f"CRITICAL: You MUST choose a tool name ONLY from this list: {allowed_names_str}.\n"
        "Output ONLY a valid JSON object in this exact format:\n"
        '{"name": "the_tool_name", "parameters": {"arg_name": "value"}}\n\n'
        f"Available tools:\n{tools_doc}"
    )
    prefix = '{"name": "'
    full_prompt = f"{system_instruction}\nUser: {prompt}\nAssistant:\n{prefix}"
    # Encoding: We get a clean list of integers using [0].tolist()
    generated_ids = model.encode(full_prompt)[0].tolist()
    state = JSONState(functions)
    state.current_step = "FUNC_NAME"
    for i in range(100):
        full_text = model.decode(generated_ids)
        current_gen_text = full_text.split("Assistant:")[-1]
        state.update_state(current_gen_text)
        #  get Logits
        raw_logits_list = model.get_logits_from_input_ids(generated_ids)
        # convert the list into a PyTorch tensor in order to filter it
        last_token_logits = torch.tensor(raw_logits_list, dtype=torch.float32)
        # apply my filtre
        filtered_logits = json_constraints(last_token_logits, state, vocab)
        # chose the high proba
        next_token_id = int(torch.argmax(filtered_logits).item())
        generated_ids.append(next_token_id)
        # Verify text with new token
        final_text = model.decode(generated_ids)
        final_gen_text = final_text.split("Assistant:")[-1].strip()
        # stop consition
        if ((next_token_id == model._tokenizer.eos_token_id) or
            (final_gen_text.count("{") == final_gen_text.count("}") and
             final_gen_text.count("{") > 0)):
            return model.decode(generated_ids).split("Assistant:")[-1].strip()

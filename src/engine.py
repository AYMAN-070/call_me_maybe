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
    logits.fill_(-float('inf'))
    def allow(token_text):
        t_id = vocab.get(token_text)
        if t_id is not None: logits[t_id] = 0
    if state.current_step == "START":
        allow("{")
    elif state.current_step == "KEY_NAME":
        allow('"name"')
    elif state.current_step == "COLON" or state.current_step == "ARG_COLON":
        allow(":")
    elif state.current_step == "FUNC_NAME":
        for fn_name in state.functions.keys():
            target = f'"{fn_name}"'
            token_id = vocab.get(target)
            if token_id is not None: 
                logits[token_id] = original_logits[token_id]
            else:
                print(f"Warning: Token {target} not found in vocab!")
    elif state.current_step == "COMMA":
        allow(",")
    elif state.current_step == "KEY_ARGS":
        allow('"parameters"')
    elif state.current_step == "ARG_START":
        allow("{")
    elif state.current_step == "VALUE":
        """Give back to IA original scores, so it can chose """
        return original_logits 
    elif state.current_step == "END":
        allow("}")
    return logits

def generate_response(model, prompt: str, functions: List[FunctionDef], vocab: Dict):
    """Generates a complete JSON response using constraint decoding."""
    tools_doc = ""
    for f in functions:
        tools_doc += f"- {f.name}: {f.description}. Parameters: {f.parameters}\n"
    system_instruction = (
        "System: You are a function calling assistant.\n"
        f"You have access to these tools:\n{tools_doc}\n"
        "Output ONLY the JSON call. No talk."
    )
    full_prompt = f"{system_instruction}\nUser: {prompt}\nAssistant:"
    # Encoding: We get a clean list of integers using [0].tolist()
    generated_ids = model.encode(full_prompt)[0].tolist()
    state = JSONState(functions)
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
        # stop consition
        if next_token_id == model.tokenizer.eos_token_id or current_gen_text.strip().endswith("}"):
            break

    return model.decode(generated_ids).split("Assistant:")[-1].strip()
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
        elif any(f'"{fn}"' in clean_text for fn in self.functions):
            self.current_step = "END"

def json_constraints(logits: torch.Tensor, state: JSONState, vocab: Dict) -> torch.Tensor:
    """Filters logits based on the logical state
    Takes scores(logits) and kill forbiden words"""
    # take a copy of IA scores
    original_logits = logits.clone()
    logits.fill_(-float('inf'))
    if state.current_step == "START":
        token_id = vocab.get("{")
        if token_id: logits[token_id] = 0
    elif state.current_step == "KEY_NAME":
        token_id = vocab.get('"name"')
        if token_id: logits[token_id] = 0
    elif state.current_step == "COLON":
        token_id = vocab.get(":")
        if token_id: logits[token_id] = 0
    elif state.current_step == "FUNC_NAME":
        for fn_name in state.functions.keys():
            target = f'"{fn_name}"'
            token_id = vocab.get(target)
            if token_id is not None: 
                logits[token_id] = original_logits[token_id]
            else:
                print(f"Warning: Token {target} not found in vocab!")
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
    #text to digits (IDS)
    input_ids = model.encode(full_prompt)
    generated_ids = List(input_ids)
    state = JSONState(functions)
    for i in range(100):
        #Convert the list to Tensor format for PyTorch
        full_text = model.decode(generated_ids)
        current_gen_text = full_text.split("Assistant:")[-1]
        state.update_state(current_gen_text)
        input_tensor = torch.tensor([generated_ids])
        #ask for probabilities
        logits = model.get_logits_from_input_ids(input_tensor)
        last_token_logits = logits[0, -1, :]
        filtered_logits = json_constraints(last_token_logits, state, vocab)
        #chosing best token(with higher score)
        next_token_id = int(torch.argmax(filtered_logits))
        generated_ids.append(next_token_id)
        #eos: end of sentence
        if next_token_id == model.tokenizer.eos_token_id or input_ids.endswith("}"):
            break
    return model.decode(generated_ids).split("Assistant:")[-1].strip()
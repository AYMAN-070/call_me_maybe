class JSONStateMachine:
    def __init__(self, allowed_functions):
        self.allowed_functions = allowed_functions
        self.current_state = "START"
        self.generated_text = ""

    def get_allowed_tokens(self, vocab):
        allowed_ids = []
        if self.current_state == "START":
            target = "{"
            allowed_ids = [i for t, i in vocab.items() if t == target]
        elif self.current_state == "KEY_NAME":
            target = '"name"'
            allowed_ids = [i for t, i in vocab.items() if t == target]
        
        return allowed_ids
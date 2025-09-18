class ChatMemory:
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.history = []

    def add_message(self, user: str, assistant: str):
        self.history.append((user, assistant))
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def get_context(self):
        context = ""
        for user, assistant in self.history:
            context += f"User: {user}\nAssistant: {assistant}\n"
        return context.strip()

    def clear(self):
        self.history = []

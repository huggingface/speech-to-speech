class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size=1):
        self.size = size
        self.init_chat_message = None
        self.buffer = []

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) > 2 * self.size:
            self.buffer = self.buffer[-2*self.size:]

    def init_chat(self, init_chat_message):
        self.init_chat_message = init_chat_message

    def to_list(self):
        context = self.buffer[-2*self.size:] if self.size > 0 else []
        return [self.init_chat_message] + context if self.init_chat_message else context

    def reset_context(self):
        self.buffer = []

    def get_last_pair(self):
        return self.buffer[-2:] if len(self.buffer) >= 2 else self.buffer
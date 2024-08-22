


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size):
        self.size = size
        self.init_chat_message = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer = []

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

    def init_chat(self, init_chat_message):
        self.init_chat_message = init_chat_message

    def to_list(self):
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer

import re

_SENTENCE_END = re.compile(r"[.!?]+(?=\s)")


class ConversationalStreamParser:
    def __init__(self):
        self.current = ""
        self.finished = False

    def feed(self, text):
        outputs = []
        self.current += text

        while True:
            match = _SENTENCE_END.search(self.current)

            if match is None:
                break

            end = match.end()
            sentence = self.current[:end].strip()
            self.current = self.current[end:].lstrip()

            if sentence:
                outputs.append(sentence)

        return outputs

    def flush(self):
        outputs = []

        if self.current.strip():
            outputs.append(self.current.strip())
            self.current = ""

        return outputs

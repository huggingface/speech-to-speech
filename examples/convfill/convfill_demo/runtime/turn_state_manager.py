class TurnStateManager:
    def __init__(self, boundary_tokens):
        self.user_turn = ""
        self.responses = []
        self.thoughts = []
        self.boundary_tokens = boundary_tokens

    def add_user_turn(self, user_turn):
        self.user_turn = user_turn

    def add_thought(self, thought):
        self.thoughts.append(thought)

    def add_response(self, response):
        self.responses.append(response)

    def reset(self):
        self.user_turn = ""
        self.responses = []
        self.thoughts = []

    def user_turn_empty(self):
        return self.user_turn == ""

    def build_prompt(self, current_user_input, current_thought, history):
        prompt = ""
        assert len(history["user_turns_history"]) == len(history["responses_history"])
        for i in range(len(history["user_turns_history"])):
            prompt += (
                self.boundary_tokens["USER_START"] + history["user_turns_history"][i] + self.boundary_tokens["END"]
            )
            prompt += self.boundary_tokens["ASSN_START"] + history["responses_history"][i] + self.boundary_tokens["END"]

        prompt += self.boundary_tokens["USER_START"] + current_user_input + self.boundary_tokens["END"]
        if len(self.responses) > 0:
            prompt += self.boundary_tokens["ASSN_START"]
            for response in self.responses:
                prompt += response + " "
            prompt += self.boundary_tokens["END"]
        prompt += self.boundary_tokens["KNOWLEDGE_START"] + current_thought + self.boundary_tokens["END"]
        prompt += self.boundary_tokens["ASSN_START"]
        return prompt

    def get_completed_response(self):
        final_response = ""
        for response in self.responses:
            final_response += response + " "
        return self.user_turn, final_response.strip(), self.thoughts

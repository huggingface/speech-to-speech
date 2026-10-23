"""Self-hosted streaming Reasoner; no commercial provider SDK is required."""

from .remote_inference import RemoteInference


class BackendInference:
    def __init__(self, api_key, model_name, model_mode, remote_config=None):
        if model_mode != "vllm":
            raise ValueError("This example requires a self-hosted vllm Reasoner")
        self.remote = RemoteInference({"model": model_name, **(remote_config or {})})

    def begin_turn(self, cancel_event):
        self.remote.begin_turn(cancel_event)

    def cancel(self):
        self.remote.cancel()

    def infer(self, prompt):
        yield from self.remote.knowledge(prompt)

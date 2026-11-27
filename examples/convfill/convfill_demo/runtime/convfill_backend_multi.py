"""Normal-conversation ConvFill Reasoner prompt adapter."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class ConvFillBackend:
    def __init__(self, dialogue_state_manager, prompt_template_path, model_backend, mode="normal", **kwargs):
        if mode != "normal":
            raise ValueError("The hosted example supports normal conversation only")
        self.dialogue_state_manager = dialogue_state_manager
        self.model_backend = model_backend
        self.prompt_template_path = Path(prompt_template_path)

    def cancel(self):
        self.model_backend.cancel()

    def begin_turn(self, cancel_event):
        self.model_backend.begin_turn(cancel_event)

    def backend_infer(self):
        env = Environment(loader=FileSystemLoader(self.prompt_template_path.parent))
        template = env.get_template(self.prompt_template_path.name)
        prompt = template.render(conversation=self.dialogue_state_manager.get_transcript())
        yield from self.model_backend.infer(prompt)

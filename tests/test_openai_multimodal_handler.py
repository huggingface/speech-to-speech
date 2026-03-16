from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.chat import Chat
from LLM.openai_api_language_model import OpenApiModelHandler


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="It looks like a desk."))]
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def make_handler():
    handler = object.__new__(OpenApiModelHandler)
    handler.model_name = "fake-vlm"
    handler.stream = False
    handler.chat = Chat(5)
    handler.user_role = "user"
    handler.image_detail = "auto"
    handler.default_images = []
    handler.client = FakeClient()
    return handler


def test_normalize_default_images_converts_local_path_to_data_url(tmp_path):
    handler = make_handler()
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-image-bytes")

    images = handler._normalize_default_images(str(image_path), None)

    assert len(images) == 1
    assert images[0]["type"] == "image_url"
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert images[0]["image_url"]["detail"] == "auto"


def test_process_builds_multimodal_user_message_with_default_image():
    handler = make_handler()
    handler.default_images = [
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/cat.jpg",
                "detail": "high",
            },
        }
    ]

    output = list(handler.process("What is in this image?"))

    assert output == [("It looks like a desk.", None, [])]
    create_call = handler.client.chat.completions.calls[0]
    user_message = create_call["messages"][0]
    assert user_message["role"] == "user"
    assert user_message["content"][0] == {
        "type": "text",
        "text": "What is in this image?",
    }
    assert user_message["content"][1]["image_url"]["url"] == "https://example.com/cat.jpg"


def test_process_supports_structured_prompt_with_extra_images_and_language():
    handler = make_handler()
    prompt = (
        {
            "text": "Describe what you see.",
            "images": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/scene.jpg",
                        "detail": "auto",
                    },
                }
            ],
        },
        "en-auto",
    )

    list(handler.process(prompt))

    create_call = handler.client.chat.completions.calls[0]
    user_message = create_call["messages"][0]
    assert user_message["content"][0]["text"].startswith(
        "Please reply to my message in english. Describe what you see."
    )
    assert user_message["content"][1]["image_url"]["url"] == "https://example.com/scene.jpg"

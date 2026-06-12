from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from queue import Queue
from threading import Event
from typing import Any, Literal

import httpx
from openai import OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent

from speech_to_speech.LLM.chat import Chat

logger = logging.getLogger(__name__)

VisionGenerateFn = Callable[[list[str], str], str]
_VisionRequestKind = Literal["user_message", "function_call_output"]

DEFAULT_VISION_QUESTION = "Describe the image for the assistant."
VISION_RESULT_LABEL = "Image analysis"


@dataclass(frozen=True)
class VisionRoutingRequest:
    kind: _VisionRequestKind
    item_id: str | None
    call_id: str | None
    question: str
    image_urls: list[str]


class VisionRouter:
    """Converts image-bearing chat items into text observations.

    Text-only LLMs can then continue the conversation without receiving image
    payloads. The router mutates the chat in place by replacing user image
    parts with an ``Image analysis: ...`` text note and by replacing camera
    ``b64_im`` tool outputs with ``image_description`` JSON.
    """

    def __init__(self, generate: VisionGenerateFn) -> None:
        self.generate = generate

    def process_chat(self, chat: Chat) -> int:
        requests = self._collect_requests(chat)
        routed = 0
        for request in requests:
            try:
                description = self.generate(request.image_urls, request.question).strip()
            except Exception as exc:
                logger.exception("Vision routing failed for %s", request.kind)
                description = f"Vision processing failed: {type(exc).__name__}: {exc}"
            if not description:
                description = "No image description was returned."
            if self._apply_observation(chat, request, description):
                routed += 1
        return routed

    def _collect_requests(self, chat: Chat) -> list[VisionRoutingRequest]:
        requests: list[VisionRoutingRequest] = []
        with chat._lock:
            function_calls = {
                item.call_id: item
                for item in chat.buffer
                if isinstance(item, RealtimeConversationItemFunctionCall) and item.call_id is not None
            }
            function_calls.update(chat._pending_tool_calls)

            for item in chat.buffer:
                if isinstance(item, RealtimeConversationItemUserMessage):
                    image_urls = [
                        p.image_url
                        for p in item.content
                        if p.type == "input_image" and p.image_url is not None
                    ]
                    if not image_urls:
                        continue
                    question = self._user_message_text(item) or DEFAULT_VISION_QUESTION
                    requests.append(
                        VisionRoutingRequest(
                            kind="user_message",
                            item_id=item.id,
                            call_id=None,
                            question=question,
                            image_urls=image_urls,
                        )
                    )
                elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                    payload = _json_object(item.output)
                    if payload is None or payload.get("image_description"):
                        continue
                    image_urls = _image_urls_from_payload(payload)
                    if not image_urls:
                        continue
                    fc = function_calls.get(item.call_id)
                    question = _question_from_function_call(fc) or _string_field(payload, "question")
                    requests.append(
                        VisionRoutingRequest(
                            kind="function_call_output",
                            item_id=item.id,
                            call_id=item.call_id,
                            question=question or DEFAULT_VISION_QUESTION,
                            image_urls=image_urls,
                        )
                    )
        return requests

    def _apply_observation(self, chat: Chat, request: VisionRoutingRequest, description: str) -> bool:
        with chat._lock:
            for item in chat.buffer:
                if request.kind == "user_message" and isinstance(item, RealtimeConversationItemUserMessage):
                    if item.id != request.item_id:
                        continue
                    current_text = self._user_message_text(item)
                    item.content = [UserContent(type="input_text", text=_merge_observation(current_text, description))]
                    return True

                if request.kind == "function_call_output" and isinstance(
                    item, RealtimeConversationItemFunctionCallOutput
                ):
                    if item.id != request.item_id and item.call_id != request.call_id:
                        continue
                    payload = _json_object(item.output)
                    if payload is None:
                        return False
                    payload.pop("b64_im", None)
                    payload.pop("b64_image", None)
                    payload.pop("image_b64", None)
                    payload["image_description"] = description
                    if request.question and "question" not in payload:
                        payload["question"] = request.question
                    item.output = json.dumps(payload, ensure_ascii=False)
                    return True
        return False

    @staticmethod
    def _user_message_text(item: RealtimeConversationItemUserMessage) -> str:
        return " ".join(p.text for p in item.content if p.type == "input_text" and p.text).strip()


class ResponsesApiVisionClient:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str | None,
        api_key: str | None,
        request_timeout_s: float,
        disable_thinking: bool,
        max_output_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.timeout = httpx.Timeout(request_timeout_s, connect=min(10.0, request_timeout_s))
        self.max_output_tokens = max_output_tokens
        self.extra_body = (
            {"chat_template_kwargs": {"enable_thinking": False}}
            if disable_thinking and base_url is not None and base_url != "https://api.openai.com/v1"
            else None
        )

    def generate(self, image_urls: list[str], question: str) -> str:
        content: list[dict[str, str]] = [
            {
                "type": "input_text",
                "text": _vision_prompt(question),
            }
        ]
        content.extend({"type": "input_image", "image_url": image_url, "detail": "auto"} for image_url in image_urls)
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Answer visual questions concisely using only the provided image evidence.",
                        }
                    ],
                },
                {"type": "message", "role": "user", "content": content},
            ],
            max_output_tokens=self.max_output_tokens,
            extra_body=self.extra_body,
            timeout=self.timeout,
        )
        return response.output_text


class LocalVisionClient:
    def __init__(
        self,
        *,
        stop_event: Event,
        backend: Literal["transformers", "mlx"],
        model_name: str,
        device: str,
        torch_dtype: str,
        gen_kwargs: dict[str, Any],
    ) -> None:
        from speech_to_speech.LLM.language_model import VisionLanguageModelHandler

        self.handler = VisionLanguageModelHandler(
            stop_event,
            queue_in=Queue(),
            queue_out=Queue(),
            setup_kwargs={
                "model_name": model_name,
                "device": device,
                "torch_dtype": torch_dtype,
                "gen_kwargs": gen_kwargs,
                "backend": backend,
                "stream_batch_sentences": 1,
                "compact_history": False,
            },
        )

    def generate(self, image_urls: list[str], question: str) -> str:
        from speech_to_speech.LLM.language_model import StreamContext

        chat = Chat(size=1)
        chat.add_item(_make_vision_user_message(question, image_urls))
        ctx = StreamContext()
        chunks = list(self.handler._generate(chat, None, None, ctx))
        return ctx.generated_text.strip() or " ".join(chunk.text for chunk in chunks).strip()


def build_vision_router(stop_event: Event, kwargs: dict[str, Any] | None) -> VisionRouter | None:
    kwargs = kwargs or {}
    backend = (kwargs.get("vision_backend") or "none").strip()
    if backend == "none":
        return None

    model_name = kwargs.get("vision_model_name")
    if not model_name:
        raise ValueError("--vision_model_name is required when --vision_backend is not 'none'.")

    max_new_tokens = int(kwargs.get("vision_gen_max_new_tokens", 256))
    if backend == "responses-api":
        client = ResponsesApiVisionClient(
            model_name=model_name,
            base_url=kwargs.get("vision_responses_api_base_url"),
            api_key=kwargs.get("vision_responses_api_api_key"),
            request_timeout_s=float(kwargs.get("vision_responses_api_request_timeout_s", 20.0)),
            disable_thinking=bool(kwargs.get("vision_responses_api_disable_thinking", True)),
            max_output_tokens=max_new_tokens,
        )
        return VisionRouter(client.generate)

    if backend in ("transformers", "mlx-lm"):
        local_backend: Literal["transformers", "mlx"] = "mlx" if backend == "mlx-lm" else "transformers"
        client = LocalVisionClient(
            stop_event=stop_event,
            backend=local_backend,
            model_name=model_name,
            device=kwargs.get("vision_device", "cuda"),
            torch_dtype=kwargs.get("vision_torch_dtype", "float16"),
            gen_kwargs={
                "max_new_tokens": max_new_tokens,
                "temperature": float(kwargs.get("vision_gen_temperature", 0.0)),
                "do_sample": bool(kwargs.get("vision_gen_do_sample", False)),
            },
        )
        return VisionRouter(client.generate)

    raise ValueError("--vision_backend must be one of: none, responses-api, transformers, mlx-lm.")


def _make_vision_user_message(question: str, image_urls: list[str]) -> RealtimeConversationItemUserMessage:
    content = [UserContent(type="input_text", text=_vision_prompt(question))]
    content.extend(UserContent(type="input_image", image_url=image_url) for image_url in image_urls)
    return RealtimeConversationItemUserMessage(type="message", role="user", content=content)


def _vision_prompt(question: str) -> str:
    return f"{question.strip()}\n\nReply with a concise visual answer for another assistant to use."


def _merge_observation(text: str, description: str) -> str:
    if text:
        return f"{text}\n\n{VISION_RESULT_LABEL}: {description}"
    return f"{VISION_RESULT_LABEL}: {description}"


def _json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _string_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""


def _image_urls_from_payload(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("image_url", "url"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            urls.append(value)
    raw_urls = payload.get("image_urls")
    if isinstance(raw_urls, list):
        urls.extend(value for value in raw_urls if isinstance(value, str) and value)

    for key in ("b64_im", "b64_image", "image_b64"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            urls.append(value if value.startswith("data:") else f"data:image/jpeg;base64,{value}")
    return urls


def _question_from_function_call(fc: RealtimeConversationItemFunctionCall | None) -> str:
    if fc is None:
        return ""
    payload = _json_object(fc.arguments)
    if payload is None:
        return ""
    return _string_field(payload, "question")

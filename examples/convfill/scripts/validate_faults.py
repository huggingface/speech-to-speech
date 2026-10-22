"""Actual Talker/Reasoner inference with transport faults from fault_proxy.py."""

import json
import os
import time
from pathlib import Path

import httpx
from convfill_demo.runtime.remote_inference import InferenceError
from convfill_demo.runtime.run_convfill import ConvFillConfig, ConvFillSystem


def main():
    proxy = os.environ.get("CONVFILL_FAULT_PROXY", "http://127.0.0.1:18766")
    config = ConvFillConfig("configs/demo_mode/convfill_hosted_config.json", mode="normal")
    config.reasoner["base_url"] = proxy + "/v1"
    config.reasoner["base_url_env"] = "CONVFILL_FAULT_REASONER_URL"
    config.reasoner["request_timeout_s"] = 8
    events, records = [], []
    start = time.perf_counter()

    def on_thought(value):
        events.append(("filler" if value == "<sil>" else "knowledge", time.perf_counter() - start))

    def on_response(value):
        events.append(("talker", time.perf_counter() - start))
        assert "<" not in value

    system = ConvFillSystem(
        config, emit_tts=False, on_thought=on_thought, on_response_fragment=on_response, audio_in_flight_fn=lambda: 0
    )
    try:
        cases = [
            ("delay", {"delay_s": 3}, False),
            ("http_error", {"http_error": True}, True),
            ("stream_interruption", {"interrupt_stream": True}, True),
            ("timeout", {"delay_s": 3}, True),
        ]
        for name, mode, fail in cases:
            httpx.post(
                proxy + "/control", json={"delay_s": 0, "http_error": False, "interrupt_stream": False, **mode}
            ).raise_for_status()
            system.reset()
            system.convfill_backend.model_backend.remote.request_timeout = 8
            if name == "timeout":
                system.convfill_backend.model_backend.remote.request_timeout = 0.4
            events.clear()
            start = time.perf_counter()
            error = None
            try:
                assert system.run_turn("Why is the sky blue?") is True
            except InferenceError as exc:
                error = str(exc)
            record = {
                "case": name,
                "elapsed_s": time.perf_counter() - start,
                "failed": error is not None,
                "history_turns": len(system.dialogue_state_manager.responses),
            }
            expected_error = {
                "http_error": "HTTP 503",
                "stream_interruption": "ended without [DONE]",
                "timeout": "deadline exceeded",
            }.get(name)
            record["passed"] = error is None if not fail else bool(error and expected_error in error)
            if error is not None:
                record["error"] = error
            if fail:
                assert not system.dialogue_state_manager.user_turns
            elif error is None:
                first_knowledge = next(t for kind, t in events if kind == "knowledge")
                first_talker = next(t for kind, t in events if kind == "talker")
                assert first_talker < first_knowledge
                record.update(
                    first_talker_s=first_talker, first_knowledge_s=first_knowledge, filler_preceded_knowledge=True
                )
            records.append(record)
            Path("/tmp/convfill-fault-validation.json").write_text(
                json.dumps({"actual_models": True, "injected_transport_faults": True, "tests": records}, indent=2)
                + "\n"
            )
            print(json.dumps(record), flush=True)
    finally:
        httpx.post(proxy + "/control", json={"delay_s": 0, "http_error": False, "interrupt_stream": False})
    assert all(record["passed"] for record in records), "Some live fault checks did not pass; inspect the saved results"


if __name__ == "__main__":
    main()

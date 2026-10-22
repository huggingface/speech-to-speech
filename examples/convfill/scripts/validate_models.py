"""Exercise actual configured models. Outputs only timings and pass/fail by default."""

import argparse
import json
import threading
import time
from pathlib import Path

from convfill_demo.runtime.run_convfill import ConvFillConfig, ConvFillSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo_mode/convfill_hosted_config.json")
    parser.add_argument("--output", default="/tmp/convfill-model-validation.json")
    parser.add_argument("--wait-ready-s", type=float, default=180)
    args = parser.parse_args()
    records, active = [], {}

    def thought(value):
        elapsed = time.perf_counter() - active["start"]
        if value == "<sil>":
            active.setdefault("first_filler_signal_s", elapsed)
        else:
            active.setdefault("first_useful_reasoner_sentence_s", elapsed)
            active["knowledge_sentences"] += 1
            if system._frontend_inflight > 0:
                active["knowledge_during_talker_generation"] = True

    def fragment(value):
        active.setdefault("first_talker_sentence_s", time.perf_counter() - active["start"])
        active["talker_sentences"] += 1
        assert "<" not in value, "control token leaked to Talker output"

    config = ConvFillConfig(args.config, mode="normal")
    system = ConvFillSystem(
        config, on_thought=thought, on_response_fragment=fragment, emit_tts=False, audio_in_flight_fn=lambda: 0
    )
    import httpx

    deadline = time.monotonic() + args.wait_ready_s
    for stage in [system.convfill_backend.model_backend.remote, system.convfill_frontend.remote]:
        while True:
            try:
                headers = {"Authorization": "Bearer " + stage.api_key} if stage.api_key else {}
                with httpx.Client(transport=httpx.HTTPTransport(retries=2), timeout=5) as client:
                    response = client.get(stage.base_url + "/models", headers=headers)
                response.raise_for_status()
                if stage.model in [m["id"] for m in response.json().get("data", [])]:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Configured model did not become ready")
            time.sleep(1)
    for prompt in [
        "My name is Alex. Why is the sky blue?",
        "What name did I tell you?",
        "Explain that sky answer in one short sentence.",
    ]:
        active = {"start": time.perf_counter(), "knowledge_sentences": 0, "talker_sentences": 0}
        completed = system.run_turn(prompt)
        active["total_s"] = time.perf_counter() - active.pop("start")
        active["completed"] = completed
        response = system.dialogue_state_manager.responses[-1].lower()
        expected = "alex" if prompt.startswith("What name") else "blue"
        active["expected_fact_in_talker_response"] = expected in response
        assert active["expected_fact_in_talker_response"], "Talker did not preserve the expected conversational fact"
        active["history_turns"] = len(system.dialogue_state_manager.responses)
        records.append(active)
        print(json.dumps(active), flush=True)
    # A real stream cancellation while the models are active, then a fresh turn.
    active = {"start": time.perf_counter(), "knowledge_sentences": 0, "talker_sentences": 0}
    errors = []

    def run():
        try:
            system.run_turn("Explain how a rainbow forms in detail.")
        except Exception as exc:
            errors.append(type(exc).__name__)

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.15)
    start_cancel = time.perf_counter()
    system.cancel_turn()
    worker.join(5)
    assert not worker.is_alive(), "cancellation did not terminate the turn"
    system.discard_pending_turn("Explain how a rainbow forms in detail.")
    cancellation = {"cancel_join_s": time.perf_counter() - start_cancel, "errors": errors}
    system.reset()
    assert system.dialogue_state_manager.responses == []
    Path(args.output).write_text(
        json.dumps(
            {"actual_inference": True, "browser_test": False, "turns": records, "cancellation": cancellation}, indent=2
        )
        + "\n"
    )
    print(json.dumps(cancellation), flush=True)


if __name__ == "__main__":
    main()

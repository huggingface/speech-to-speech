"""Integration tests for the session-scoped voices routes.

Drives the full FastAPI app via ``create_app`` with a fake pipeline unit
(mirroring test_websocket_router.py). The voices routes are the capability
probe for voice cloning: they answer 200 when a voice store is wired
(Qwen3-TTS backend), 409 otherwise, and 404 for unknown sessions.
"""

from queue import Queue
from threading import Event as ThreadingEvent

from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.voice_store import MAX_UPLOAD_BYTES, VoiceStore
from tests.wav_utils import wav_bytes as _wav_bytes


def _make_unit(index: int = 0) -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    return PipelineUnit(
        index=index,
        service=RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen),
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


def _upload(name: str = "My Voice", ref_text: str = "hello reference", audio: bytes | None = None) -> dict:
    return {
        "files": {"audio": ("clip.wav", audio if audio is not None else _wav_bytes(), "audio/wav")},
        "data": {"ref_text": ref_text, "name": name},
    }


class TestCapabilityGating:
    def test_get_answers_409_without_voice_store(self):
        app = create_app(pool=[_make_unit()], stop_event=ThreadingEvent())
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.get(f"/v1/realtime/sessions/{session_id}/voices")
                assert r.status_code == 409
                assert r.json()["error"]["code"] == "voice_cloning_unsupported"

    def test_post_answers_409_without_voice_store(self):
        app = create_app(pool=[_make_unit()], stop_event=ThreadingEvent())
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                kw = _upload()
                r = client.post(f"/v1/realtime/sessions/{session_id}/voices", **kw)
                assert r.status_code == 409

    def test_unknown_session_answers_404(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        app = create_app(pool=[_make_unit()], stop_event=ThreadingEvent(), voice_store=store)
        with TestClient(app) as client:
            r = client.get("/v1/realtime/sessions/sess_nope/voices")
            assert r.status_code == 404
            assert r.json()["error"]["code"] == "unknown_session"


class TestVoicesRoutes:
    def _client_and_session(self, tmp_path):
        store = VoiceStore(tmp_path / "voices")
        app = create_app(pool=[_make_unit()], stop_event=ThreadingEvent(), voice_store=store)
        return app, store

    def test_get_returns_empty_list_on_fresh_store(self, tmp_path):
        app, _ = self._client_and_session(tmp_path)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.get(f"/v1/realtime/sessions/{session_id}/voices")
                assert r.status_code == 200
                assert r.json() == {"voices": []}

    def test_post_creates_voice_and_get_lists_it(self, tmp_path):
        app, _ = self._client_and_session(tmp_path)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.post(f"/v1/realtime/sessions/{session_id}/voices", **_upload())
                assert r.status_code == 201
                created = r.json()
                assert created["name"] == "My Voice"
                assert created["voice_id"]

                listed = client.get(f"/v1/realtime/sessions/{session_id}/voices").json()
                assert [v["voice_id"] for v in listed["voices"]] == [created["voice_id"]]
                assert listed["voices"][0]["name"] == "My Voice"
                assert "ref_text" not in listed["voices"][0]

    def test_duplicate_upload_returns_same_voice_id(self, tmp_path):
        app, _ = self._client_and_session(tmp_path)
        audio = _wav_bytes()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                first = client.post(f"/v1/realtime/sessions/{session_id}/voices", **_upload(audio=audio))
                second = client.post(f"/v1/realtime/sessions/{session_id}/voices", **_upload(audio=audio))
                assert first.status_code == second.status_code == 201
                assert first.json()["voice_id"] == second.json()["voice_id"]
                listed = client.get(f"/v1/realtime/sessions/{session_id}/voices").json()
                assert len(listed["voices"]) == 1

    def test_oversized_upload_rejected_with_413(self, tmp_path):
        app, _ = self._client_and_session(tmp_path)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.post(
                    f"/v1/realtime/sessions/{session_id}/voices",
                    **_upload(audio=b"\x00" * (MAX_UPLOAD_BYTES + 1)),
                )
                assert r.status_code == 413
                assert r.json()["error"]["code"] == "audio_too_large"

    def test_reupload_with_corrected_transcript_overwrites_it(self, tmp_path):
        app, store = self._client_and_session(tmp_path)
        audio = _wav_bytes()
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                url = f"/v1/realtime/sessions/{session_id}/voices"
                first = client.post(url, **_upload(audio=audio, ref_text="typo transcript"))
                second = client.post(url, **_upload(audio=audio, ref_text="fixed transcript"))
                assert first.json()["voice_id"] == second.json()["voice_id"]
                resolved = store.resolve(first.json()["voice_id"])
                assert resolved is not None
                assert resolved.ref_text == "fixed transcript"

    def test_validation_errors_map_to_status_codes(self, tmp_path):
        app, _ = self._client_and_session(tmp_path)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                url = f"/v1/realtime/sessions/{session_id}/voices"

                r = client.post(url, **_upload(audio=b"not audio"))
                assert r.status_code == 415
                assert r.json()["error"]["code"] == "audio_unreadable"

                r = client.post(url, **_upload(audio=_wav_bytes(seconds=1.0)))
                assert r.status_code == 400
                assert r.json()["error"]["code"] == "audio_too_short"

                r = client.post(url, **_upload(name="   "))
                assert r.status_code == 400
                assert r.json()["error"]["code"] == "invalid_name"

    def test_hub_push_failure_maps_to_502(self, tmp_path):
        """A voice the fleet cannot see must fail the upload (rolled back store-side)."""

        class _DownHub:
            def revision(self):
                return "rev-0"

            def list_voice_ids(self, revision=None):
                return set()

            def download_voice(self, voice_id, root, revision=None):
                raise AssertionError("nothing to download")

            def upload_voice(self, root, voice_id):
                raise RuntimeError("hub unreachable")

        store = VoiceStore(tmp_path / "voices", hub=_DownHub())
        app = create_app(pool=[_make_unit()], stop_event=ThreadingEvent(), voice_store=store)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
                r = client.post(f"/v1/realtime/sessions/{session_id}/voices", **_upload())
                assert r.status_code == 502
                assert r.json()["error"]["code"] == "voice_store_sync_failed"
                listed = client.get(f"/v1/realtime/sessions/{session_id}/voices").json()
                assert listed["voices"] == []

    def test_post_to_released_session_answers_404(self, tmp_path):
        app, _ = self._client_and_session(tmp_path)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                session_id = ws.receive_json()["session"]["id"]
            # Session disconnected (drain may be pending); voices routes must
            # treat it as gone.
            r = client.get(f"/v1/realtime/sessions/{session_id}/voices")
            assert r.status_code == 404

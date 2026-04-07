import asyncio
import logging
import json
from queue import Empty

from pipeline_control import SESSION_END, is_control_message

logger = logging.getLogger(__name__)


class WebSocketStreamer:
    """
    Handles bidirectional audio streaming over WebSocket.

    Receives audio from clients and puts it in the input_queue.
    Sends audio from the output_queue to clients.
    Sends text messages (transcripts/tools) from text_output_queue to clients.
    """

    def __init__(
        self,
        stop_event,
        input_queue,
        output_queue,
        should_listen,
        text_output_queue=None,
        host="0.0.0.0",
        port=8765,
    ):
        self.stop_event = stop_event
        self.input_queue = input_queue  # clients -> VAD
        self.output_queue = output_queue  # TTS -> clients
        self.text_output_queue = text_output_queue  # Text messages -> clients
        self.should_listen = should_listen
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.server = None

    def run(self):
        """Run the WebSocket server (called from a thread)."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self.loop.close()

    async def _run_server(self):
        """Main async server loop."""
        import websockets

        logger.info(f"WebSocket server starting on ws://{self.host}:{self.port}")

        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )

        logger.info("WebSocket server ready, waiting for connections...")

        # Start the sender task
        sender_task = asyncio.create_task(self._send_loop())

        # Wait until stop_event is set
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)

        # Cleanup
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        # Close all clients
        for client in list(self.clients):
            try:
                await client.close()
            except Exception:
                pass

        self.server.close()
        await self.server.wait_closed()
        logger.info("WebSocket server closed")

    async def _handle_client(self, websocket):
        """Handle a single WebSocket client connection."""
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")
        self.clients.add(websocket)
        recv_buffer = bytearray()

        # Enable listening when first client connects
        if len(self.clients) == 1:
            # Drain edge queues so stale data from a previous session doesn't
            # leak into the new one (SESSION_END may not have flushed everything).
            for q in (self.output_queue, self.text_output_queue):
                if q is not None:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except Empty:
                            break
            self.should_listen.set()
            logger.debug("Listening enabled, edge queues drained (should_listen.set())")

        try:
            logger.debug(f"Client {client_id}: Starting message receive loop")
            async for message in websocket:
                if isinstance(message, bytes):
                    logger.debug(f"Client {client_id}: Received {len(message)} bytes of audio")
                    if self.should_listen.is_set():
                        # Split into 512-sample (1024 bytes) chunks for VAD.
                        # Keep a per-client remainder buffer so no samples are dropped
                        # when WebSocket frame boundaries are not aligned.
                        chunk_size_bytes = 512 * 2  # 512 samples * 2 bytes per int16
                        recv_buffer.extend(message)
                        num_chunks = 0
                        while len(recv_buffer) >= chunk_size_bytes:
                            chunk = bytes(recv_buffer[:chunk_size_bytes])
                            del recv_buffer[:chunk_size_bytes]
                            self.input_queue.put(chunk)
                            num_chunks += 1
                        logger.debug(f"Client {client_id}: Queued {num_chunks} chunks for processing")
                    else:
                        logger.debug(f"Client {client_id}: Skipping audio (should_listen not set)")

        except Exception as e:
            logger.error(f"Client {client_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected (finally block)")

            if len(self.clients) == 0:
                logger.debug("Last WebSocket client disconnected, ending session")
                self.input_queue.put(SESSION_END)

    async def _send_loop(self):
        """Send audio and text from queues to all connected clients."""
        # Buffer audio until we have at least 100ms worth (3200 bytes = 1600 samples at 16kHz int16)
        MIN_AUDIO_BYTES = 3200
        audio_buffer = bytearray()

        while not self.stop_event.is_set():
            try:
                # Check for audio
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                        if audio_buffer and self.clients:
                            data = bytes(audio_buffer)
                            audio_buffer.clear()
                            await asyncio.gather(
                                *[client.send(data) for client in self.clients],
                                return_exceptions=True,
                            )
                        break
                    if is_control_message(audio_chunk, SESSION_END.kind):
                        audio_buffer.clear()
                        continue

                    if self.clients:
                        if hasattr(audio_chunk, 'tobytes'):
                            audio_chunk = audio_chunk.tobytes()
                        audio_buffer.extend(audio_chunk)

                        if len(audio_buffer) >= MIN_AUDIO_BYTES:
                            data = bytes(audio_buffer)
                            audio_buffer.clear()
                            logger.debug(f"Sending {len(data)} bytes of audio to {len(self.clients)} client(s)")
                            await asyncio.gather(
                                *[client.send(data) for client in self.clients],
                                return_exceptions=True
                            )
                except Empty:
                    # Flush any buffered audio when queue is empty
                    if audio_buffer and self.clients:
                        data = bytes(audio_buffer)
                        audio_buffer.clear()
                        logger.debug(f"Flushing {len(data)} bytes of audio to {len(self.clients)} client(s)")
                        await asyncio.gather(
                            *[client.send(data) for client in self.clients],
                            return_exceptions=True
                        )

                # Check for text/tool messages
                if self.text_output_queue:
                    try:
                        text_message = self.text_output_queue.get_nowait()
                        if self.clients:
                            # Send as JSON string
                            await asyncio.gather(
                                *[client.send(json.dumps(text_message)) for client in self.clients],
                                return_exceptions=True
                            )
                    except Empty:
                        pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

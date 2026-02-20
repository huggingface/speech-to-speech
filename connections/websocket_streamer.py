import asyncio
import logging
import json
from queue import Empty

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

        # Enable listening when first client connects
        if len(self.clients) == 1:
            self.should_listen.set()
            logger.debug(f"Listening enabled (should_listen.set())")

        try:
            logger.debug(f"Client {client_id}: Starting message receive loop")
            async for message in websocket:
                if isinstance(message, bytes):
                    logger.debug(f"Client {client_id}: Received {len(message)} bytes of audio")
                    if self.should_listen.is_set():
                        # Split into 512-sample (1024 bytes) chunks for VAD
                        chunk_size_bytes = 512 * 2  # 512 samples * 2 bytes per int16
                        num_chunks = 0
                        for i in range(0, len(message), chunk_size_bytes):
                            chunk = message[i:i + chunk_size_bytes]
                            # Only queue complete 512-sample chunks
                            if len(chunk) == chunk_size_bytes:
                                self.input_queue.put(chunk)
                                num_chunks += 1
                            else:
                                logger.debug(f"Client {client_id}: Skipping incomplete chunk ({len(chunk)} bytes)")
                        logger.debug(f"Client {client_id}: Queued {num_chunks} chunks for processing")
                    else:
                        logger.debug(f"Client {client_id}: Skipping audio (should_listen not set)")

        except Exception as e:
            logger.error(f"Client {client_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected (finally block)")

            if len(self.clients) == 0:
                self.input_queue.put(b"END")

    async def _send_loop(self):
        """Send audio and text from queues to all connected clients."""
        while not self.stop_event.is_set():
            try:
                # Check for audio
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                        break

                    if self.clients:
                        if hasattr(audio_chunk, 'tobytes'):
                            audio_chunk = audio_chunk.tobytes()
                        logger.info(f"Sending {len(audio_chunk)} bytes of audio to {len(self.clients)} client(s)")
                        await asyncio.gather(
                            *[client.send(audio_chunk) for client in self.clients],
                            return_exceptions=True
                        )
                except Empty:
                    pass

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

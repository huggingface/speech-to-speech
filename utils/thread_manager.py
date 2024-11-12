import threading
import logging

logger = logging.getLogger(__name__)

class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run)
            self.threads.append(thread)
            thread.start()
            logger.debug(f'Thread {thread.ident} has started. Target: {thread.name}, type: {type(handler)}')

    def join_all(self): 
        for thread in self.threads:
            logger.debug(f'Thread {thread.ident} attempt to join. Target: {thread.name}')
            # Allow the main thread to remain responsive to KeyboardInterrupt while waiting for a child thread to finish
            while thread.is_alive():
              thread.join(timeout=0.5)
            logger.debug(f'Thread {thread.ident} has joined. Target: {thread.name}')

    def stop(self):
        logger.debug("Server stop was invoked")
        for handler in self.handlers:
            handler.stop()
        self.join_all()

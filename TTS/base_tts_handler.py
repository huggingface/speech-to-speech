from baseHandler import BaseHandler

class BaseTTSHandler(BaseHandler):
    def __init__(self, stop_event, queue_in, queue_out, setup_args=(), setup_kwargs={}, aec_handler=None):
        super().__init__(stop_event, queue_in, queue_out, setup_args, setup_kwargs)
        self.aec_handler = aec_handler
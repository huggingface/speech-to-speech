"""Read endpoint credentials on the server only."""

import os


def get_endpoint_api_key(env_var):
    return os.environ.get(env_var, "") if env_var else ""

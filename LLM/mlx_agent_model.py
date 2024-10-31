import logging
from LLM.chat import Chat
from baseHandler import BaseHandler
from mlx_lm import load, stream_generate, generate
from rich.console import Console
import torch

logger = logging.getLogger(__name__)

console = Console()

from transformers.agents import CodeAgent, tool
from datetime import datetime
import random
from huggingface_hub import list_models
from mlx_lm import load, stream_generate
import webbrowser
import pyperclip
import requests

@tool
def get_random_number_between(min: int, max: int) -> int:
    """
    Gets a random number between 1 and 100.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return random.randint(min, max)


@tool
def get_weather(city: str) -> str:
    """
    Returns the weather forecast for a given city.

    Args:
        city: The name of the city.

    Returns:
        A string with a mock weather forecast.
    """
    url = 'https://wttr.in/{}?format=+%C,+%t'.format(city)
    res = requests.get(url).text

    return f"The weather in {city} is {res.split(',')[0]} with a high of {res.split(',')[1][:-2]} degrees Celsius."

@tool
def get_current_time() -> str:
    """
    This is a tool that returns the current time.
    It returns the current time as HH:MM.
    """
    return f"The current time is {datetime.now().hour}:{datetime.now().minute}."

@tool
def open_webbrowser(url: str) -> str:
    """
    This is a tool that opens a web browser to the given website.

    Args:
        url: The url to open.
    """
    webbrowser.open(url)
    return f"I opened {url.replace('https://', '').replace('www.', '')} in the browser."


@tool
def summarize_clipboard_content() -> str:
    """
    Extracts content from the clipboard and summarizes it using another LLM (mocked for now).

    Returns:
        A string containing the summarized text.
    """
    # Extracting the clipboard content
    clipboard_content = pyperclip.paste()

    # Mocking the call to another LLM for summarization
    def mock_llm_summarize(text: str) -> str:
        # Simulating summarization
        return f"Summary: {text[:50]}..."  # Truncate for demo purposes
    
    # Check if clipboard is empty
    if not clipboard_content.strip():
        return "Clipboard is empty or has unsupported content."
    
    # Summarizing the clipboard content
    summary = mock_llm_summarize(clipboard_content)
    return summary



from jinja2 import Template

system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<<tool_descriptions>>

<<managed_agents_descriptions>>

You can use imports in your code, but only from the following list of modules: <<authorized_imports>>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>"""


import json
from transformers.agents import CodeAgent


import re
def parse_response(text: str) -> str | dict[str, any]:
    """Parses a response from the model, returning either the
    parsed list with the tool calls parsed, or the
    model thought or response if couldn't generate one.

    Args:
        text: Response from the model.
    """
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return json.loads(matches[0])
    return text


# manager_agent = CodeAgent(tools=tools, llm_engine=llm_engine, system_prompt=system_prompt)

# manager_agent.run("What time is it?")

# manager_agent.run(
#     "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
# )
# manager_agent.run("Can you give me the current hour?")
# manager_agent.run("Can you give me the current minute?")
# manager_agent.run("Can you give me a random number between the current hour and minute?")


WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}

class MLXAgentModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="smol-explorers/SmolLM2-1.7B-Intermediate-SFT-v2-lr1e5-FC",
        device="mps",
        torch_dtype="float16",
        gen_kwargs={},
    ):
        self.mlx_model, self.mlx_tokenizer = load(model_name)

        self.model_name = model_name
        self.gen_kwargs = gen_kwargs

        self.tools = [
            get_current_time,
            get_random_number_between,
            open_webbrowser,
            get_weather,
            summarize_clipboard_content
        ]

        self.toolbox = {tool.name: tool for tool in self.tools}
        self.json_code_agent = CodeAgent(tools=self.tools, llm_engine=self.llm_engine, system_prompt=system_prompt)

    def llm_engine(self,messages, stop_sequences=["Task", "<|endoftext|>"]) -> str:
        prompt = self.mlx_tokenizer.apply_chat_template(messages, tokenize=False)
        output = ""
        for t in stream_generate(self.mlx_model, self.mlx_tokenizer, prompt=prompt):
            output += t
            if t in stop_sequences:
                break
        return output
    

    def call_tools(self, tool_calls: list[dict[str, any]]) -> list[any]:
        tool_responses = []
        for tool_call in tool_calls:
            if tool_call["name"] in self.toolbox:
                tool_responses.append(self.toolbox[tool_call["name"]](*tool_call["arguments"].values()))
            else:
                tool_responses.append(f"Tool {tool_call['name']} not found.")
        return tool_responses

    def process(self, prompt):
        logger.debug("infering language model...")
        language_code = None

        if isinstance(prompt, tuple):
            prompt, language_code = prompt

        response = self.json_code_agent.run(prompt, return_generated_code=True)
        print("Response:")
        print(response)
        tool_calls = parse_response(response)
        print("Tool calls:")
        print(tool_calls)
        tool_responses = self.call_tools(tool_calls)
        print("Tool responses:")
        print(tool_responses)
        for tool_response in tool_responses:
            yield (tool_response, language_code)
        torch.mps.empty_cache()

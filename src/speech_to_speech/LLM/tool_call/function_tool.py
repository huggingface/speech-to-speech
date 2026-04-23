import textwrap

from openai.types.realtime import RealtimeFunctionTool

from speech_to_speech.LLM.tool_call.signature_from_schema import signature_from_schema


class FunctionTool(RealtimeFunctionTool):
    def to_code_prompt(self, include_args_doc: bool = True) -> str:
        """Generate a code-style prompt string for this function tool.

        Args:
            include_args_doc: If True, include argument descriptions in the docstring.
                ⚠️ This lets the model see each argument's purpose but significantly increases
                token usage (e.g. 906 tokens without vs 3434 with for the default Reachy Mini
                tool profile). Enable depending on the model's capabilities and context limit.
        """
        signature = signature_from_schema(self.parameters)

        tool_doc = self.description or ""

        if isinstance(self.parameters, dict) and include_args_doc:
            props = self.parameters.get("properties", {})
            if props:
                arg_lines = []
                for arg_name, arg_schema in props.items():
                    desc = arg_schema.get("description", "") if isinstance(arg_schema, dict) else ""
                    arg_lines.append(f"{arg_name}: {desc}")
                args_doc = f"Args:\n{textwrap.indent(chr(10).join(arg_lines), '    ')}"
                tool_doc += f"\n\n{args_doc}"

        tool_doc = f'"""{tool_doc}\n"""'

        return f"def {self.name}{signature}:\n{textwrap.indent(tool_doc, '    ')}"

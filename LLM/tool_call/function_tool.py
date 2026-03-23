import textwrap
from LLM.tool_call.signature_from_schema import signature_from_schema
from openai.types.realtime import RealtimeFunctionTool

class FunctionTool(RealtimeFunctionTool):

    def to_code_prompt(self) -> str:
        signature = signature_from_schema(self.parameters)

        tool_doc = self.description or ""

        if isinstance(self.parameters, dict):
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
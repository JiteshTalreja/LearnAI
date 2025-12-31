from langchain_core.tools import render_text_description

from langchain_core.tools import tool


@tool
def some_function(something1: str, something2: str) -> int:
    """This is a sample function to demonstrate rendering text description."""
    pass

print((render_text_description([some_function])))
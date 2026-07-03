"""
STEP 4: The actual AI AGENT - uses Groq's free hosted API + its tool-calling
feature directly (no heavy framework needed).

This shows the full "agentic" loop:
  1. Send the question + list of available tools to the LLM
  2. LLM decides which tool(s) to call and with what arguments
  3. We actually run those Python functions
  4. We send the results back to the LLM
  5. LLM writes the final plain-English answer

Setup:
  pip install groq
  Get a free key at https://console.groq.com
  Set it as an environment variable before running (see README).
Run:
  python agent_groq.py
"""

import os
import json
from groq import Groq

from agent_tools import get_sales_by_region, compare_month_over_month, detect_big_drops

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
        "and set it as an environment variable before running again."
    )

client = Groq(api_key=api_key)

# NOTE: Groq periodically retires older models. If this model name ever
# stops working, check https://console.groq.com/docs/models for the
# current recommended replacement.
MODEL = "openai/gpt-oss-20b"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sales_by_region",
            "description": "Get total units sold and revenue for a region, optionally filtered by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "e.g. North, South, East, West, Central"},
                    "category": {"type": "string", "description": "e.g. Shampoo, Detergent, Soap (optional)"},
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_month_over_month",
            "description": "Compare a given month's sales to the previous month for a region and category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "category": {"type": "string"},
                    "year": {"type": "integer"},
                    "month": {"type": "integer", "description": "1 to 12"},
                },
                "required": ["region", "category", "year", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_big_drops",
            "description": "Scan all region+category combinations and flag any month-over-month sales drop bigger than the threshold percent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold_pct": {"type": "number", "description": "Default 30"},
                },
                "required": [],
            },
        },
    },
]

available_functions = {
    "get_sales_by_region": get_sales_by_region,
    "compare_month_over_month": compare_month_over_month,
    "detect_big_drops": detect_big_drops,
}

SYSTEM_PROMPT = (
    "You are a helpful retail sales analyst assistant talking to a business "
    "client (not a developer). Answer in simple, clear English. Always use "
    "the tools to fetch real data before answering - never guess numbers. "
    "If you find a big drop, briefly suggest a possible reason and offer to "
    "notify the data team."
)


def ask_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024,
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )

        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1024,
        )
        return second_response.choices[0].message.content

    return response_message.content


if __name__ == "__main__":
    print("Ask-Your-Sales-Data Agent [Groq backend] (type 'exit' to quit)")
    while True:
        q = input("\nYou: ")
        if q.lower() == "exit":
            break
        print("\nAgent:", ask_agent(q))
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

from agent_tools import (
    get_sales_by_region,
    compare_month_over_month,
    detect_big_drops,
    get_highest_selling_in_week,
    compare_products_week_over_week,
    detect_declining_products,
)

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
    {
        "type": "function",
        "function": {
         "name": "get_highest_selling_in_week",
            "description": "Find the top-selling category and the top product within that category for a region within a specific date range (e.g. one week, Feb 1 2024 to Feb 8 2024). Also returns the single highest-selling product overall, even if it belongs to a different category. If the user names a specific category (e.g. 'in the Shampoo category'), pass it in the category field to get the top product within just that category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "e.g. North, South, East, West, Central"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                    "category": {"type": "string", "description": "Optional - e.g. Shampoo, Detergent, Soap. Only set this if the user specifically names a category."},
                },
                "required": ["region", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products_week_over_week",
            "description": "Compare a category's sales in a specific week against the week right before it, for a region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "category": {"type": "string"},
                    "week_start_date": {"type": "string", "description": "The week's date in YYYY-MM-DD format"},
                },
                "required": ["region", "category", "week_start_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_declining_products",
            "description": "Scan all products week by week and flag any product that has been declining for several consecutive weeks in a row. Optionally filter to one region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Optional - leave out to scan all regions"},
                    "min_consecutive_weeks": {"type": "integer", "description": "Default 5"},
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
    "get_highest_selling_in_week": get_highest_selling_in_week,
    "compare_products_week_over_week": compare_products_week_over_week,
    "detect_declining_products": detect_declining_products,
}

SYSTEM_PROMPT = """
You are an AI Retail Sales Assistant helping a business client understand and improve their retail sales.

RULES FOR FACTS AND NUMBERS:
- Always use the available tools to fetch real sales data before stating any specific number, product name, or trend.
- Never invent or guess a specific figure, product, or comparison that wasn't returned by a tool.

RULES FOR SUGGESTIONS / RECOMMENDATIONS:
- If the user asks for suggestions to improve sales in a region/category, FIRST call a relevant tool
  (compare_month_over_month, detect_big_drops, or detect_declining_products) for that region/category
  to check whether there is a real, data-backed decline or drop.
- Then give 2-4 short, practical, general retail business suggestions (e.g. running a promotion, checking
  competitor pricing, reviewing stock availability, seasonal demand planning).
- Clearly label these as general suggestions based on common retail practice, not guaranteed outcomes,
  since truly improving sales depends on business context beyond this dataset.
- If the tool shows no clear negative trend, say so honestly, and still offer general suggestions for
  maintaining or growing sales further.

If the user's request needs data or analysis with absolutely no matching tool (e.g. comparing many regions
at once in ways no tool supports, or something unrelated to retail sales), respond:
"I'm sorry, I can't answer that with the data and tools currently available. Please contact your Data Analyst or Data Engineering team for further analysis."

Only make the tool calls you actually need. After receiving tool result(s), generate your final answer using
those results (for facts) plus general business knowledge (for suggestions, clearly labeled as such).
"""

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

    if not tool_calls:
        return (
            "I'm sorry, I can't answer that with the data and tools currently "
            "available. Please contact your Data Analyst or Data Engineering team "
            "for further analysis."
        )

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

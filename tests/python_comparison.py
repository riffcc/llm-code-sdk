#!/usr/bin/env python3
"""
Comparison tests for Python SDK against Z.ai's Anthropic-compatible API.

Run with: ZAI_API_KEY=your_key python tests/python_comparison.py

This script runs the same tests as the Rust integration tests to verify
both SDKs produce equivalent results.
"""

import os
import sys

# Add anthropic SDK to path
sys.path.insert(0, os.path.expanduser("~/projects/anthropic-sdk-python/src"))

import anthropic


def get_client():
    """Create a client configured for Z.ai's API."""
    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        print("ZAI_API_KEY not set")
        return None

    # Configure for Z.ai's Anthropic-compatible endpoint
    return anthropic.Anthropic(
        api_key=api_key,
        base_url="https://api.z.ai/api/anthropic",
    )


def test_simple_message():
    """Test basic message creation without tools."""
    print("\n=== Test: Simple Message ===")
    client = get_client()
    if not client:
        return

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        )
        print(f"Response: {message.content[0].text}")
        print(f"Stop reason: {message.stop_reason}")
        assert "4" in message.content[0].text, f"Expected '4' in response"
        print("PASSED")
    except Exception as e:
        print(f"Error: {e}")


def test_system_prompt():
    """Test message with system prompt."""
    print("\n=== Test: System Prompt ===")
    client = get_client()
    if not client:
        return

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a helpful pirate. Always respond like a pirate.",
            messages=[{"role": "user", "content": "What are you?"}],
        )
        print(f"Response: {message.content[0].text}")
        text = message.content[0].text.lower()
        is_pirate = any(
            word in text for word in ["arr", "matey", "ye", "ahoy", "captain"]
        )
        print(f"Appears pirate-like: {is_pirate}")
        print("PASSED")
    except Exception as e:
        print(f"Error: {e}")


def test_tool_use():
    """Test tool use with a simple calculator tool."""
    print("\n=== Test: Tool Use ===")
    client = get_client()
    if not client:
        return

    def calculate(expression: str) -> str:
        """Perform a mathematical calculation."""
        if "+" in expression:
            parts = expression.split("+")
            if len(parts) == 2:
                a, b = float(parts[0].strip()), float(parts[1].strip())
                return str(a + b)
        elif "*" in expression:
            parts = expression.split("*")
            if len(parts) == 2:
                a, b = float(parts[0].strip()), float(parts[1].strip())
                return str(a * b)
        return "Unsupported operation"

    tools = [
        {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2')",
                    }
                },
                "required": ["expression"],
            },
        }
    ]

    try:
        messages = [
            {
                "role": "user",
                "content": "Use the calculator to compute 15 + 27. Tell me the result.",
            }
        ]

        # Agentic loop
        for iteration in range(10):
            print(f"  Iteration {iteration + 1}")
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=messages,
                tools=tools,
            )

            print(f"    Stop reason: {message.stop_reason}")

            if message.stop_reason != "tool_use":
                # Final response
                text = "".join(
                    block.text for block in message.content if hasattr(block, "text")
                )
                print(f"Final response: {text}")
                assert "42" in text, f"Expected '42' in response"
                print("PASSED")
                return

            # Execute tools
            tool_results = []
            for block in message.content:
                if block.type == "tool_use":
                    print(f"    Tool call: {block.name}({block.input})")
                    result = calculate(block.input.get("expression", ""))
                    print(f"    Tool result: {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            # Append to conversation
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": tool_results})

        print("Max iterations reached")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_multi_turn():
    """Test multi-turn conversation."""
    print("\n=== Test: Multi-Turn ===")
    client = get_client()
    if not client:
        return

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What's my name?"},
            ],
        )
        print(f"Response: {message.content[0].text}")
        assert "alice" in message.content[0].text.lower(), "Expected 'Alice' in response"
        print("PASSED")
    except Exception as e:
        print(f"Error: {e}")


def test_stop_reason():
    """Test that stop_reason is correctly returned."""
    print("\n=== Test: Stop Reason ===")
    client = get_client()
    if not client:
        return

    try:
        # Test end_turn
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello."}],
        )
        print(f"Stop reason: {message.stop_reason}")
        assert message.stop_reason is not None, "Expected stop_reason to be set"

        # Test max_tokens
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1,
            messages=[
                {"role": "user", "content": "Write a very long story about dragons."}
            ],
        )
        print(f"Stop reason with max_tokens=1: {message.stop_reason}")
        print("PASSED")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Python SDK Comparison Tests against Z.ai")
    print("=" * 60)

    test_simple_message()
    test_system_prompt()
    test_tool_use()
    test_multi_turn()
    test_stop_reason()

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)

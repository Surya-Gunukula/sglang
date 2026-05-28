"""Unit tests for DeepSeekV3Detector — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestDeepSeekV3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV3Detector()

    def _make_tool_call(self, name, args_dict):
        args_json = json.dumps(args_dict)
        return (
            f'<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n'
            f'```json\n{args_json}\n```<｜tool▁call▁end｜>'
        )

    def _wrap_calls(self, *calls):
        return "<｜tool▁calls▁begin｜>" + "\n".join(calls) + "<｜tool▁calls▁end｜>"

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        text = self._wrap_calls(self._make_tool_call("get_weather", {"city": "Tokyo"}))
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is nice today."))

    # ==================== detect_and_parse ====================

    def test_single_tool_call(self):
        text = self._wrap_calls(
            self._make_tool_call("get_weather", {"city": "Beijing"})
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_multiple_tool_calls(self):
        text = self._wrap_calls(
            self._make_tool_call("get_weather", {"city": "Tokyo"}),
            self._make_tool_call("search", {"query": "restaurants"}),
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_leading_text(self):
        call = self._make_tool_call("get_weather", {"city": "Paris"})
        text = "Let me check that for you. " + self._wrap_calls(call)
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Let me check that for you.")

    def test_no_tool_call(self):
        result = self.detector.detect_and_parse("Just regular text.", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "Just regular text.")

    def test_multiple_arguments(self):
        text = self._wrap_calls(
            self._make_tool_call("get_weather", {"city": "London", "unit": "celsius"})
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    # ==================== structure_info ====================

    def test_structure_info(self):
        info_fn = self.detector.structure_info()
        info = info_fn("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("tool▁calls▁begin", info.trigger)
        self.assertIn("tool▁call▁end", info.end)

    # ==================== streaming ====================

    def test_streaming_single_tool_call(self):
        chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function",
            '<｜tool▁sep｜>get_weather\n```json\n{"city": ',
            '"Tokyo"}\n```',
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]
        all_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        names = [c.name for c in all_calls if c.name is not None]
        self.assertIn("get_weather", names)
        arg_parts = [c.parameters for c in all_calls if c.parameters]
        full_args = "".join(arg_parts)
        self.assertIn("Tokyo", full_args)

    def test_streaming_multiple_calls_sequential_indices(self):
        """Regression: greedy regex caused the first call to be skipped when
        multiple complete tool calls arrived in the buffer simultaneously."""
        call1 = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Tokyo"}\n```<｜tool▁call▁end｜>'
        )
        call2 = (
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            '```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜>'
        )
        full = f"<｜tool▁calls▁begin｜>{call1}\n{call2}<｜tool▁calls▁end｜>"

        all_calls = []
        result = self.detector.parse_streaming_increment(full, self.tools)
        all_calls.extend(result.calls)
        for _ in range(5):
            result = self.detector.parse_streaming_increment("", self.tools)
            all_calls.extend(result.calls)
            if not result.calls:
                break

        name_calls = [c for c in all_calls if c.name is not None]
        param_calls = [c for c in all_calls if c.parameters]
        self.assertEqual(len(name_calls), 2, "Both calls should be detected")
        self.assertEqual(name_calls[0].tool_index, 0)
        self.assertEqual(name_calls[1].tool_index, 1)
        all_params = "".join(c.parameters for c in param_calls)
        self.assertIn("Tokyo", all_params)
        self.assertIn("Paris", all_params)

    def test_streaming_no_tool_call(self):
        result = self.detector.parse_streaming_increment("Hello world", self.tools)
        self.assertEqual(result.normal_text, "Hello world")
        self.assertEqual(len(result.calls), 0)


if __name__ == "__main__":
    unittest.main()

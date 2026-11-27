"""Tests for the Gemini integration. A fake client and a patched network call are
used, so these tests need no API key and make no real requests."""
import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from agents.llm_agent import LLMAgent
from api.gemini_client import GeminiClient
from api.prompts import move_prompt
from game.board import Board


class FakeClient:
    def __init__(self, reply, configured=True):
        self.reply = reply
        self.is_configured = configured
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def gemini_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class LLMAgentTests(unittest.TestCase):
    def test_reads_a_json_move(self):
        self.assertEqual(LLMAgent(FakeClient('{"move": 4}')).choose_move(Board.empty()), 4)

    def test_falls_back_to_the_first_number_in_plain_text(self):
        self.assertEqual(LLMAgent(FakeClient("I would play 7")).choose_move(Board.empty()), 7)

    def test_sends_a_prompt_to_the_client(self):
        client = FakeClient('{"move": 0}')
        LLMAgent(client).choose_move(Board.empty())
        self.assertEqual(len(client.prompts), 1)
        self.assertTrue(client.prompts[0])

    def test_occupied_square_is_rejected(self):
        board = Board.empty().apply_move(4)
        with self.assertRaises(RuntimeError):
            LLMAgent(FakeClient('{"move": 4}')).choose_move(board)

    def test_square_off_the_board_is_rejected(self):
        with self.assertRaises(RuntimeError):
            LLMAgent(FakeClient('{"move": 12}')).choose_move(Board.empty())

    def test_reply_without_a_number_is_rejected(self):
        with self.assertRaises(RuntimeError):
            LLMAgent(FakeClient("no idea, sorry")).choose_move(Board.empty())

    def test_non_integer_move_is_rejected(self):
        with self.assertRaises(RuntimeError):
            LLMAgent(FakeClient('{"move": "centre"}')).choose_move(Board.empty())

    def test_availability_follows_the_client_configuration(self):
        self.assertTrue(LLMAgent(FakeClient("{}", configured=True)).is_available)
        self.assertFalse(LLMAgent(FakeClient("{}", configured=False)).is_available)


class PromptTests(unittest.TestCase):
    def test_prompt_is_text_that_asks_for_a_move(self):
        prompt = move_prompt(Board.empty())
        self.assertIsInstance(prompt, str)
        self.assertIn("move", prompt.lower())

    def test_prompt_describes_the_position(self):
        board = Board.empty().apply_move(4)          
        prompt = move_prompt(board)
        self.assertIn('["_", "_", "_", "_", "X", "_", "_", "_", "_"]', prompt)
        self.assertIn("Your mark: O", prompt)
        self.assertIn("[0, 1, 2, 3, 5, 6, 7, 8]", prompt)     
        self.assertIn("3 x 3", prompt)

    def test_prompt_adapts_to_board_size(self):
        self.assertIn("4 x 4", move_prompt(Board.empty(4)))


class GeminiClientTests(unittest.TestCase):
    def test_is_not_configured_without_a_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = GeminiClient()
        self.assertFalse(client.is_configured)

    def test_generate_without_a_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = GeminiClient()
        with self.assertRaises(RuntimeError):
            client.generate("hello")

    def test_key_and_model_can_come_from_the_environment(self):
        env = {"GEMINI_API_KEY": "abc", "GEMINI_MODEL": "test-model"}
        with mock.patch.dict(os.environ, env, clear=True):
            client = GeminiClient()
        self.assertTrue(client.is_configured)
        self.assertEqual(client.model, "test-model")

    def test_default_model(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "abc"}, clear=True):
            self.assertEqual(GeminiClient().model, "gemini-2.0-flash")

    @mock.patch("api.gemini_client.urllib.request.urlopen")
    def test_returns_the_text_of_the_reply(self, urlopen):
        urlopen.return_value = FakeResponse(gemini_reply('{"move": 3}'))
        self.assertEqual(GeminiClient(api_key="k").generate("hi"), '{"move": 3}')

    @mock.patch("api.gemini_client.urllib.request.urlopen")
    def test_request_carries_the_prompt_and_json_settings(self, urlopen):
        urlopen.return_value = FakeResponse(gemini_reply("ok"))
        GeminiClient(api_key="k", model="m").generate("my prompt")
        request = urlopen.call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "my prompt")
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
        self.assertAlmostEqual(body["generationConfig"]["temperature"], 0.1)
        self.assertIn("m:generateContent", request.full_url)

    @mock.patch("api.gemini_client.urllib.request.urlopen")
    def test_http_errors_become_runtime_errors(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "http://x", 429, "Too Many Requests", {}, io.BytesIO(b"quota exceeded"))
        with self.assertRaises(RuntimeError) as caught:
            GeminiClient(api_key="k").generate("hi")
        self.assertIn("429", str(caught.exception))

    @mock.patch("api.gemini_client.urllib.request.urlopen")
    def test_network_errors_become_runtime_errors(self, urlopen):
        urlopen.side_effect = urllib.error.URLError("no network")
        with self.assertRaises(RuntimeError):
            GeminiClient(api_key="k").generate("hi")

    @mock.patch("api.gemini_client.urllib.request.urlopen")
    def test_unexpected_reply_format_is_rejected(self, urlopen):
        urlopen.return_value = FakeResponse({"candidates": []})
        with self.assertRaises(RuntimeError):
            GeminiClient(api_key="k").generate("hi")


if __name__ == "__main__":
    unittest.main()
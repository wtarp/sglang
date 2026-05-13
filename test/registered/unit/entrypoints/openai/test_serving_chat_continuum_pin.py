import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _contains_request_stream(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == "stream":
            if isinstance(sub.value, ast.Name) and sub.value.id == "request":
                return True
    return False


class TestServingChatContinuumPin(unittest.TestCase):
    def test_non_streaming_does_not_send_continuum_pin(self):
        repo_root = Path(__file__).resolve().parents[5]
        serving_chat_path = (
            repo_root / "python/sglang/srt/entrypoints/openai/serving_chat.py"
        )
        self.assertTrue(serving_chat_path.exists())

        module = ast.parse(serving_chat_path.read_text(encoding="utf-8"))

        build_fn: ast.FunctionDef | None = None
        for node in module.body:
            if not isinstance(node, ast.ClassDef) or node.name != "OpenAIServingChat":
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_build_chat_response":
                    build_fn = item
                    break
        self.assertIsNotNone(build_fn)
        assert build_fn is not None

        pin_ifs: list[ast.If] = []
        for item in ast.walk(build_fn):
            if not isinstance(item, ast.If):
                continue
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "ContinuumPinReqInput"
                for call in ast.walk(item)
            ):
                pin_ifs.append(item)

        guarded_pin_ifs = [node for node in pin_ifs if _contains_request_stream(node.test)]
        self.assertEqual(len(guarded_pin_ifs), 1)

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "python/sglang").exists():
            return parent
    raise AssertionError("Could not locate repo root from test file path")


def _is_self_call(stmt: ast.stmt, *, attr: str) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and func.attr == attr
    )


class TestSchedulerContinuumPinnedWaitingQueueAST(unittest.TestCase):
    def test_calc_priority_calls_promote_pinned_waiting_queue(self):
        repo_root = _find_repo_root(Path(__file__).resolve())
        scheduler_path = repo_root / "python/sglang/srt/managers/scheduler.py"
        self.assertTrue(scheduler_path.exists())

        module = ast.parse(scheduler_path.read_text(encoding="utf-8"))

        scheduler_cls: ast.ClassDef | None = None
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
                scheduler_cls = node
                break
        self.assertIsNotNone(scheduler_cls)
        assert scheduler_cls is not None

        has_helper = any(
            isinstance(item, ast.FunctionDef)
            and item.name == "_continuum_promote_pinned_waiting_queue"
            for item in scheduler_cls.body
        )
        self.assertTrue(has_helper)

        raw_fn: ast.FunctionDef | None = None
        for item in scheduler_cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "_get_new_batch_prefill_raw":
                raw_fn = item
                break
        self.assertIsNotNone(raw_fn)
        assert raw_fn is not None

        idx_calc = None
        idx_promote = None
        for idx, stmt in enumerate(raw_fn.body):
            if _is_self_call(stmt, attr="_continuum_promote_pinned_waiting_queue"):
                idx_promote = idx
            elif (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and isinstance(stmt.value.func.value, ast.Attribute)
                and isinstance(stmt.value.func.value.value, ast.Name)
                and stmt.value.func.value.value.id == "self"
                and stmt.value.func.value.attr == "policy"
                and stmt.value.func.attr == "calc_priority"
            ):
                idx_calc = idx

        self.assertIsNotNone(idx_calc)
        self.assertIsNotNone(idx_promote)
        assert idx_calc is not None
        assert idx_promote is not None
        self.assertGreater(idx_promote, idx_calc)

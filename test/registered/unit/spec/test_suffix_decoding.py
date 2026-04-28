import sys
import types
import unittest
from unittest.mock import Mock, patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")

_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestSuffixServerArgs(CustomTestCase):
    @patch("sglang.srt.server_args.is_arctic_inference_available", return_value=False)
    def test_missing_arctic_inference_raises(self, _mock_arctic):
        with self.assertRaises(ImportError):
            ServerArgs(
                model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                speculative_algorithm="SUFFIX",
            )

    @patch("sglang.srt.server_args.is_arctic_inference_available", return_value=True)
    def test_suffix_defaults_num_draft_tokens_from_tree_depth(self, _mock_arctic):
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            speculative_algorithm="SUFFIX",
            speculative_suffix_max_tree_depth=32,
        )

        self.assertEqual(server_args.speculative_num_draft_tokens, 32)
        self.assertTrue(server_args.disable_overlap_schedule)
        self.assertFalse(server_args.enable_mixed_chunk)

    @patch("sglang.srt.server_args.is_arctic_inference_available", return_value=True)
    def test_suffix_rejects_invalid_ranges(self, _mock_arctic):
        invalid_cases = [
            {"speculative_suffix_max_tree_depth": 0},
            {"speculative_suffix_max_cached_requests": -2},
            {"speculative_suffix_max_spec_factor": -0.1},
            {"speculative_suffix_min_token_prob": 1.1},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ServerArgs(
                        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                        speculative_algorithm="SUFFIX",
                        **kwargs,
                    )


class TestSuffixSpecAlgorithm(CustomTestCase):
    def test_suffix_algorithm_is_registered(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        suffix_algo = SpeculativeAlgorithm.from_string("SUFFIX")

        self.assertTrue(suffix_algo.is_suffix())
        self.assertFalse(suffix_algo.is_ngram())
        self.assertFalse(suffix_algo.supports_spec_v2())


class TestSuffixCacheAdapter(CustomTestCase):
    def test_batch_get_tracks_request_state_and_shapes(self):
        fake_module = types.ModuleType("arctic_inference")
        fake_suffix_module = types.ModuleType("arctic_inference.suffix_decoding")

        class FakeDraft:
            token_ids = [11, 12]
            parents = [-1, 0]

        class FakeSuffixDecodingCache:
            def __init__(self, **_kwargs):
                self.active_requests = set()
                self.started = []
                self.responses = []

            def start_request(self, request_id, prompt):
                self.active_requests.add(request_id)
                self.started.append((request_id, list(prompt)))

            def add_active_response(self, request_id, new_tokens):
                self.responses.append((request_id, list(new_tokens)))

            def speculate(self, *_args, **_kwargs):
                return FakeDraft()

            def stop_request(self, request_id):
                self.active_requests.discard(request_id)

        fake_suffix_module.SuffixDecodingCache = FakeSuffixDecodingCache

        with patch.dict(
            sys.modules,
            {
                "arctic_inference": fake_module,
                "arctic_inference.suffix_decoding": fake_suffix_module,
            },
        ):
            from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter

            adapter = SuffixCacheAdapter(
                draft_token_num=4,
                max_batch_size=2,
                max_tree_depth=8,
            )
            draft_tokens, tree_mask = adapter.batch_get(
                ["req-1"],
                [[1, 2]],
                [[1, 2, 3]],
            )

        self.assertEqual(draft_tokens.shape, (4,))
        self.assertEqual(tree_mask.shape, (16,))
        self.assertEqual(draft_tokens[0], 3)


class TestSuffixWorker(CustomTestCase):
    def test_suffix_worker_uses_suffix_cache_adapter(self):
        mock_server_args = Mock()
        mock_server_args.speculative_num_draft_tokens = 16
        mock_server_args.speculative_suffix_max_tree_depth = 24
        mock_server_args.speculative_suffix_max_cached_requests = 10000
        mock_server_args.speculative_suffix_max_spec_factor = 1.0
        mock_server_args.speculative_suffix_min_token_prob = 0.1

        mock_target_worker = Mock()
        mock_target_worker.max_running_requests = 48
        mock_target_worker.model_runner = Mock()

        with patch(
            "sglang.srt.speculative.ngram_worker.NGRAMWorker.__init__",
            autospec=True,
            return_value=None,
        ) as mock_ngram_init, patch(
            "sglang.srt.speculative.suffix_worker.SuffixCacheAdapter"
        ) as mock_adapter:
            from sglang.srt.speculative.suffix_worker import SuffixWorker

            worker = SuffixWorker(
                server_args=mock_server_args,
                gpu_id=0,
                tp_rank=0,
                dp_rank=None,
                moe_ep_rank=0,
                nccl_port=12345,
                target_worker=mock_target_worker,
            )

        mock_ngram_init.assert_called_once()
        mock_adapter.assert_called_once()
        self.assertIs(worker.ngram_cache, mock_adapter.return_value)


if __name__ == "__main__":
    unittest.main()

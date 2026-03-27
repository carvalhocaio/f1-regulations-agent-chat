import unittest

from f1_agent.fine_tuning.live_dataset import (
    build_sft_examples_from_failures,
    normalize_failure_cases,
    redact_text,
    split_examples,
)


class LiveFineTuningTests(unittest.TestCase):
    def test_redact_text_masks_common_sensitive_patterns(self):
        text = (
            "email: user@example.com phone: +1 650-555-1234 "
            "url: https://example.com/a/b id: 123456789 "
            "api_key=ABCDEFGH12345678"
        )
        redacted = redact_text(text)

        self.assertIn("<REDACTED_EMAIL>", redacted)
        self.assertIn("<REDACTED_PHONE>", redacted)
        self.assertIn("<REDACTED_URL>", redacted)
        self.assertIn("<REDACTED_ID>", redacted)
        self.assertIn("<REDACTED_SECRET>", redacted)

    def test_normalize_failure_cases_deduplicates(self):
        rows = [
            {
                "prompt": "Who won 2023?",
                "expected_behavior": "Use database tool and answer Max Verstappen.",
                "failure_type": "tool_routing",
                "category": "tool_routing",
            },
            {
                "prompt": "Who won 2023?",
                "expected_behavior": "Use database tool and answer Max Verstappen.",
                "failure_type": "tool_routing",
                "category": "tool_routing",
            },
        ]

        curated = normalize_failure_cases(rows)
        self.assertEqual(len(curated), 1)
        self.assertTrue(curated[0]["id"].startswith("live-"))
        self.assertTrue(curated[0]["pii_redacted"])

    def test_build_sft_examples_from_failures_shape(self):
        cases = [
            {
                "prompt": "Who won the Brazilian GP in 2012?",
                "expected_behavior": "Call query_f1_history_template and answer Jenson Button.",
            }
        ]

        examples = build_sft_examples_from_failures(cases)
        self.assertEqual(len(examples), 1)
        ex = examples[0]
        self.assertEqual(list(ex.keys()), ["contents"])
        self.assertEqual(ex["contents"][0]["role"], "user")
        self.assertEqual(ex["contents"][1]["role"], "model")

    def test_split_examples_keeps_non_empty_train_and_test_when_possible(self):
        examples = [{"contents": [1]} for _ in range(10)]
        train, test = split_examples(examples, test_ratio=0.2, seed=1)

        self.assertEqual(len(train) + len(test), 10)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)


if __name__ == "__main__":
    unittest.main()

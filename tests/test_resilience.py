import time
import unittest

from f1_agent.resilience import (
    CircuitBreakerOpenError,
    CircuitBreakerSettings,
    RetrySettings,
    backoff_delay_seconds,
    classify_transient_error,
    is_quota_or_unavailable_error,
    run_with_retry,
)


class ResilienceUtilsTests(unittest.TestCase):
    def test_classify_transient_http_429(self):
        transient, status_code, error_type = classify_transient_error(
            Exception("429 Too Many Requests")
        )
        self.assertTrue(transient)
        self.assertEqual(status_code, 429)
        self.assertEqual(error_type, "Exception")

    def test_classify_non_transient_error(self):
        transient, status_code, _ = classify_transient_error(
            Exception("Validation failed")
        )
        self.assertFalse(transient)
        self.assertIsNone(status_code)

    def test_quota_or_unavailable_helper(self):
        self.assertTrue(is_quota_or_unavailable_error(Exception("ResourceExhausted")))
        self.assertTrue(
            is_quota_or_unavailable_error(Exception("503 Service Unavailable"))
        )
        self.assertFalse(is_quota_or_unavailable_error(Exception("400 bad request")))

    def test_backoff_delay_without_jitter_randomness(self):
        delay = backoff_delay_seconds(
            attempt=3,
            initial=0.5,
            exp_base=2.0,
            max_delay=10.0,
            jitter=0.0,
        )
        self.assertEqual(delay, 2.0)


class RetryExecutionTests(unittest.TestCase):
    def test_run_with_retry_recovers_after_transient_failures(self):
        attempts = {"count": 0}
        slept: list[float] = []

        def fn():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise Exception("503 Service Unavailable")
            return "ok"

        result = run_with_retry(
            f"test-recover-{time.time_ns()}",
            fn,
            retry=RetrySettings(
                enabled=True,
                max_attempts=3,
                initial_delay_seconds=0.1,
                max_delay_seconds=0.2,
                exp_base=2.0,
                jitter=0.0,
            ),
            circuit=CircuitBreakerSettings(
                enabled=False,
                failure_threshold=5,
                open_seconds=30.0,
            ),
            sleep_fn=lambda value: slept.append(value),
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(len(slept), 2)

    def test_circuit_breaker_opens_after_transient_failures(self):
        op = f"test-circuit-{time.time_ns()}"

        def fail_fn():
            raise Exception("429 Too Many Requests")

        retry = RetrySettings(
            enabled=True,
            max_attempts=1,
            initial_delay_seconds=0.01,
            max_delay_seconds=0.01,
            exp_base=2.0,
            jitter=0.0,
        )
        circuit = CircuitBreakerSettings(
            enabled=True,
            failure_threshold=2,
            open_seconds=60.0,
        )

        with self.assertRaises(Exception):
            run_with_retry(op, fail_fn, retry=retry, circuit=circuit)
        with self.assertRaises(Exception):
            run_with_retry(op, fail_fn, retry=retry, circuit=circuit)

        with self.assertRaises(CircuitBreakerOpenError):
            run_with_retry(op, fail_fn, retry=retry, circuit=circuit)


if __name__ == "__main__":
    unittest.main()

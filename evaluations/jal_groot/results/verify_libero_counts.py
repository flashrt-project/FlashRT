"""Recompute the published ten-round LIBERO-10 totals without model dependencies."""

import json
from pathlib import Path
import statistics


def verify(path):
    data = json.loads(Path(path).read_text())
    rounds = data["rounds"]
    assert len(rounds) == 10
    assert len({r["seed"] for r in rounds}) == 10
    assert [r["round"] for r in rounds] == list(range(1, 11))
    assert data["episodes_per_round_per_backend"] == 500
    assert data["rounds_per_backend"] == 10
    assert data["total_episodes"] == 15000
    assert set(data["aggregate"]) == {"torch", "trt", "flashrt"}
    for backend, published in data["aggregate"].items():
        counts = [r["successes"][backend] for r in rounds]
        assert all(type(n) is int and 0 <= n <= 500 for n in counts)
        rates = [n / 5 for n in counts]
        expected = dict(
            episodes=5000,
            successes=sum(counts),
            success_percent=sum(counts) / 50,
            run_to_run_sample_std_pp=statistics.stdev(rates),
        )
        assert set(expected) == set(published)
        for key, value in expected.items():
            assert abs(published[key] - value) < 1e-10, (backend, key)
        print(
            f"{backend}: {sum(counts)}/5000; {expected['success_percent']:.2f}% "
            f"+/- {expected['run_to_run_sample_std_pp']:.2f} pp"
        )
    print("PASS: all ten rounds included; totals and sample SD verified")


if __name__ == "__main__":
    verify(Path(__file__).with_name("libero-ten-rounds.json"))

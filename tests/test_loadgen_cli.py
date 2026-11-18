import json
from pathlib import Path

import httpx
import pytest

from bench.loadgen import main, sanitize_argv
from bench.results_io import read_results
from tests.fake_server import SseServer, echo_handler

SMALL = [
    "--prompt-median", "8", "--prompt-sigma", "0", "--prompt-max", "16",
    "--output-median", "4", "--output-sigma", "0", "--output-max", "8",
]  # fmt: skip


def run_cli(tmp_path: Path, name: str, *extra: str, rate: str = "200") -> tuple[int, Path]:
    out = tmp_path / name
    argv = [
        "run", "--url", "http://secret-host.internal:8000", "--model", "demo-model",
        "--rate", rate, "--num-requests", "20", "--warmup", "1", "--out", str(out), *SMALL, *extra,
    ]  # fmt: skip
    code = main(argv, transport=httpx.MockTransport(echo_handler()))
    return code, out


def test_run_writes_a_complete_result_file(tmp_path: Path) -> None:
    server_env = tmp_path / "env.json"
    server_env.write_text(json.dumps({"gpus": [{"name": "Test GPU"}]}))
    code, out = run_cli(tmp_path, "run.jsonl", "--label", "unit", "--server-env", str(server_env))
    assert code == 0

    meta, results, summary = read_results(out)
    assert len(results) == 20 and all(r.status == "ok" for r in results)
    assert summary is not None and summary["num_ok"] == 20
    assert meta["label"] == "unit" and meta["model"] == "demo-model"
    assert meta["workload"]["rate"] == 200.0 and meta["workload"]["num_requests"] == 20
    assert meta["server_env"] == {"gpus": [{"name": "Test GPU"}]}
    assert meta["client_env"]["git"] is not None and "command" not in meta["client_env"]
    assert meta["request_params"]["ignore_eos"] is True


def test_result_files_do_not_contain_the_server_address(tmp_path: Path) -> None:
    _, out = run_cli(tmp_path, "run.jsonl")
    text = out.read_text()
    assert "secret-host" not in text
    assert "--url <redacted>" in json.loads(text.splitlines()[0])["command"]


def test_missing_server_env_warns_and_records_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_cli(tmp_path, "run.jsonl")
    assert "must not be published" in capsys.readouterr().err
    meta, _, _ = read_results(out)
    assert meta["server_env"] is None


def test_refuses_to_overwrite_a_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(tmp_path, "run.jsonl")[0] == 0
    original = (tmp_path / "run.jsonl").read_text()
    code, _ = run_cli(tmp_path, "run.jsonl")
    assert code == 2 and "never overwritten" in capsys.readouterr().err
    assert (tmp_path / "run.jsonl").read_text() == original


def test_exit_code_is_nonzero_when_every_request_fails(tmp_path: Path) -> None:
    out = tmp_path / "down.jsonl"
    argv = [
        "run", "--url", "http://x", "--model", "m", "--rate", "200", "--num-requests", "5",
        "--warmup", "0", "--out", str(out), *SMALL,
    ]  # fmt: skip
    code = main(argv, transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    assert code == 1
    _, results, summary = read_results(out)
    assert all(r.error == "http_503" for r in results)
    assert summary is not None and summary["failure_counts"] == {"http_503": 5}


def test_failed_warmup_aborts_without_writing_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bad.jsonl"
    argv = ["run", "--url", "http://x", "--model", "m", "--rate", "5", "--num-requests", "5",
            "--warmup", "2", "--out", str(out), *SMALL]  # fmt: skip
    code = main(argv, transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    assert code == 2 and "warm-up" in capsys.readouterr().err
    assert not out.exists()


def test_summarize_and_plot_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, first = run_cli(tmp_path, "rate-a.jsonl", "--label", "unit", rate="100")
    _, second = run_cli(tmp_path, "rate-b.jsonl", "--label", "unit", rate="200")
    capsys.readouterr()

    assert main(["summarize", str(first)]) == 0
    assert "20/20 ok" in capsys.readouterr().out

    plots = tmp_path / "plots"
    assert main(["plot", str(first), str(second), "--out-dir", str(plots)]) == 0
    assert {p.name for p in plots.iterdir()} == {"rate-a.png", "rate-b.png", "sweep.png"}


def test_sanitize_argv_hides_url_and_home_directory() -> None:
    home = str(Path.home())
    cleaned = sanitize_argv(["--url", "http://10.0.0.5:8000", "--out", f"{home}/results/x.jsonl"])
    assert cleaned == ["--url", "<redacted>", "--out", "~/results/x.jsonl"]
    assert sanitize_argv(["--url=http://host", "--rate", "3"]) == [
        "--url=<redacted>",
        "--rate",
        "3",
    ]


def test_end_to_end_over_a_real_socket(tmp_path: Path) -> None:
    """Real HTTP, chunked streaming and timing: first token after ~50 ms, then one per ~20 ms."""
    out = tmp_path / "socket.jsonl"
    with SseServer(first_token_delay=0.05, token_delay=0.02) as url:
        code = main(
            [
                "run",
                "--url",
                url,
                "--model",
                "m",
                "--rate",
                "20",
                "--num-requests",
                "10",
                "--warmup",
                "1",
                "--out",
                str(out),
                *SMALL,
            ]
        )
    assert code == 0
    _, results, summary = read_results(out)
    assert len(results) == 10 and all(r.status == "ok" and r.output_tokens == 4 for r in results)
    assert all(r.ttft_s is not None and r.ttft_s >= 0.04 for r in results)
    assert summary is not None
    assert summary["itl_s"]["p50"] >= 0.015
    assert summary["generator_saturated"] is False

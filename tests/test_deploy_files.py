"""The deployment files (Dockerfiles, compose, Kubernetes) agree with the code they run.

None of these files has been built or applied (no Docker or cluster has been available), so these
tests check what can be checked without them: that every argument list parses with the servers'
real argument parsers, that names, ports and routes agree across files, that the two kustomize
overlays do what they claim (a small JSON-patch applier reproduces the patches), and that the
router image really does not need PyTorch.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

import bench.loadgen  # noqa: E402
import engine.server  # noqa: E402
import router.server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "deploy" / "docker"
K8S = ROOT / "deploy" / "k8s"
IMAGE_TAG = "0.1.0"


def load_all(path: Path) -> list[dict[str, Any]]:
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
    return [d for d in docs if isinstance(d, dict)]


def base_docs() -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for name in ("namespace", "models-pvc", "engine", "router"):
        for doc in load_all(K8S / "base" / f"{name}.yaml"):
            docs[f"{doc['kind']}/{doc['metadata']['name']}"] = doc
    return docs


def engine_container(statefulset: dict[str, Any]) -> dict[str, Any]:
    containers: list[dict[str, Any]] = statefulset["spec"]["template"]["spec"]["containers"]
    return containers[0]


# -- a tiny JSON-patch applier, for the operations the overlays use ------------------------------


def apply_json_patch(document: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    for op in operations:
        parts = [p.replace("~1", "/").replace("~0", "~") for p in op["path"].split("/")[1:]]
        parent: Any = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = parts[-1]
        if op["op"] == "add":
            if isinstance(parent, list):
                parent.append(op["value"]) if key == "-" else parent.insert(int(key), op["value"])
            else:
                parent[key] = op["value"]
        elif op["op"] == "replace":
            if isinstance(parent, list):
                parent[int(key)] = op["value"]
            else:
                assert key in parent, f"replace of a missing key {key}"
                parent[key] = op["value"]
        elif op["op"] == "remove":
            if isinstance(parent, list):
                del parent[int(key)]
            else:
                del parent[key]
        else:  # pragma: no cover - the overlays use only these three
            raise AssertionError(op["op"])
    return result


def overlay_patches(name: str) -> list[dict[str, Any]]:
    kustomization = yaml.safe_load((K8S / "overlays" / name / "kustomization.yaml").read_text())
    return list(kustomization["patches"])


def patched(overlay: str, kind: str, name: str) -> dict[str, Any]:
    """The base document ``kind/name`` with the overlay's JSON patches for it applied."""
    document = base_docs()[f"{kind}/{name}"]
    for patch in overlay_patches(overlay):
        target = patch.get("target")
        if target and target.get("kind") == kind and target.get("name") == name:
            document = apply_json_patch(document, yaml.safe_load(patch["patch"]))
    return document


def patched_engine(overlay: str) -> dict[str, Any]:
    return patched(overlay, "StatefulSet", "engine")


def test_the_json_patch_applier_does_what_it_says() -> None:
    doc = {"a": {"b": [1, 2, 3], "c": 1}}
    out = apply_json_patch(
        doc,
        [
            {"op": "add", "path": "/a/b/-", "value": 4},
            {"op": "replace", "path": "/a/c", "value": 9},
            {"op": "remove", "path": "/a/b/0"},
            {"op": "add", "path": "/a/d", "value": "x"},
        ],
    )
    assert out == {"a": {"b": [2, 3, 4], "c": 9, "d": "x"}}
    assert doc == {"a": {"b": [1, 2, 3], "c": 1}}  # the input is not touched
    with pytest.raises(AssertionError):
        apply_json_patch(doc, [{"op": "replace", "path": "/a/zzz", "value": 1}])


# -- the arguments parse with the real parsers ---------------------------------------------------


def parse_engine(args: list[str]) -> Any:
    return engine.server.build_parser().parse_args(args)


def parse_router(args: list[str]) -> Any:
    return router.server.build_parser().parse_args(args)


@pytest.mark.parametrize("overlay", [None, "kernel-graphs", "mock"])
def test_engine_arguments_parse_and_listen_on_every_interface(overlay: str | None) -> None:
    statefulset = base_docs()["StatefulSet/engine"] if overlay is None else patched_engine(overlay)
    args = parse_engine(engine_container(statefulset)["args"])
    assert args.host == "0.0.0.0"  # the default is localhost only, unreachable from outside the pod
    assert args.port == 8000
    assert (args.backend == "mock") == (overlay == "mock")
    if overlay != "mock":
        assert args.model_dir == Path("/models/Llama-3.2-1B-Instruct")
    assert args.paged_attention_kernel == (overlay == "kernel-graphs")
    assert args.cuda_graphs == (overlay == "kernel-graphs")


def test_router_arguments_parse_and_agree_with_the_engine_replicas() -> None:
    docs = base_docs()
    statefulset = docs["StatefulSet/engine"]
    service = docs["Service/engine"]
    engine_args = parse_engine(engine_container(statefulset)["args"])
    router_args = parse_router(
        docs["Deployment/router"]["spec"]["template"]["spec"]["containers"][0]["args"]
    )

    assert statefulset["spec"]["serviceName"] == service["metadata"]["name"] == "engine"
    assert service["spec"]["clusterIP"] == "None"  # headless, so engine-N.engine resolves
    expected = [
        f"http://{statefulset['metadata']['name']}-{n}.{statefulset['spec']['serviceName']}:{engine_args.port}"
        for n in range(statefulset["spec"]["replicas"])
    ]
    assert router_args.replica == expected
    assert router_args.host == "0.0.0.0" and router_args.port == 9000
    assert router_args.policy == "precise-prefix"
    assert router_args.load_cap <= engine_args.max_batch_size  # a cap above the batch never bites
    assert all(url.startswith("http://") for url in router_args.replica)


def test_service_ports_point_at_the_container_ports() -> None:
    docs = base_docs()
    engine_port = engine_container(docs["StatefulSet/engine"])["ports"][0]
    assert engine_port["name"] == "http" and engine_port["containerPort"] == 8000
    assert docs["Service/engine"]["spec"]["ports"][0]["targetPort"] == "http"
    router_port = docs["Deployment/router"]["spec"]["template"]["spec"]["containers"][0]["ports"][0]
    assert router_port["name"] == "http" and router_port["containerPort"] == 9000
    assert docs["Service/router"]["spec"]["ports"][0]["targetPort"] == "http"


def test_selectors_match_the_pod_labels() -> None:
    docs = base_docs()
    for kind, name, service in (
        ("StatefulSet", "engine", "Service/engine"),
        ("Deployment", "router", "Service/router"),
    ):
        workload = docs[f"{kind}/{name}"]
        labels = workload["spec"]["template"]["metadata"]["labels"]
        assert workload["spec"]["selector"]["matchLabels"].items() <= labels.items()
        assert docs[service]["spec"]["selector"].items() <= labels.items()


def test_every_probe_hits_a_route_the_server_really_has() -> None:
    from router.app import create_router_app
    from router.policies import make_policy
    from tests.test_engine_server import make_app

    engine_app, _ = make_app()
    router_app = create_router_app(["http://engine-0.engine:8000"], make_policy("round-robin"))
    routes = {
        "engine": {getattr(r, "path", "") for r in engine_app.routes},
        "router": {getattr(r, "path", "") for r in router_app.routes},
    }
    docs = base_docs()
    containers = {
        "engine": engine_container(docs["StatefulSet/engine"]),
        "router": docs["Deployment/router"]["spec"]["template"]["spec"]["containers"][0],
    }
    for name, container in containers.items():
        for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
            if probe not in container:
                continue
            spec = container[probe]
            if "httpGet" in spec:
                assert spec["httpGet"]["path"] in routes[name], (name, probe)
                assert spec["httpGet"]["port"] == "http"
            else:
                assert spec["tcpSocket"]["port"] == "http", (name, probe)
    assert "/health" in routes["engine"] and "/health" in routes["router"]
    # The router is ready only when a backend answers (its /health asks), but alive as long as it
    # accepts connections: a liveness check on /health would restart it while the engines load.
    router = containers["router"]
    assert router["readinessProbe"]["httpGet"]["path"] == "/health"
    assert "tcpSocket" in router["livenessProbe"] and "httpGet" not in router["livenessProbe"]


# -- the overlays do what they claim -------------------------------------------------------------


def test_the_base_engine_asks_for_one_gpu_and_the_weights_volume() -> None:
    container = engine_container(base_docs()["StatefulSet/engine"])
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert any(m["name"] == "models" and m["readOnly"] for m in container["volumeMounts"])


def test_the_mock_overlay_needs_no_gpu_and_no_weights() -> None:
    statefulset = patched_engine("mock")
    container = engine_container(statefulset)
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert all(m["name"] != "models" for m in container["volumeMounts"])
    assert all(v["name"] != "models" for v in statefulset["spec"]["template"]["spec"]["volumes"])
    # the PersistentVolumeClaim of the weights is deleted too, so nothing is left dangling
    deleted = [p for p in overlay_patches("mock") if "$patch: delete" in p["patch"]]
    assert len(deleted) == 1 and "pagedserve-models" in deleted[0]["patch"]


def test_the_kernel_overlay_keeps_the_gpu_and_only_adds_the_two_flags() -> None:
    patched = patched_engine("kernel-graphs")
    base = base_docs()["StatefulSet/engine"]
    added = engine_container(patched)["args"][len(engine_container(base)["args"]) :]
    assert added == ["--paged-attention-kernel", "--cuda-graphs"]
    assert engine_container(patched)["resources"] == engine_container(base)["resources"]


def test_every_overlay_points_at_the_base() -> None:
    for name in ("mock", "kernel-graphs", "single-gpu"):
        kustomization = yaml.safe_load((K8S / "overlays" / name / "kustomization.yaml").read_text())
        assert kustomization["resources"] == ["../../base"]
    base = yaml.safe_load((K8S / "base" / "kustomization.yaml").read_text())
    for resource in base["resources"]:
        assert (K8S / "base" / resource).exists()
    assert {i["name"]: i["newTag"] for i in base["images"]} == {
        "pagedserve-engine": IMAGE_TAG,
        "pagedserve-router": IMAGE_TAG,
    }


# -- the containers ------------------------------------------------------------------------------


def dockerfile(name: str) -> str:
    return (DOCKER / f"Dockerfile.{name}").read_text(encoding="utf-8")


def instruction(text: str, keyword: str) -> list[str]:
    """The value of every instruction ``keyword`` (continuation lines joined, comments dropped)."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    joined = re.sub(r"\\\n\s*", " ", "\n".join(lines))
    return [m.group(1).strip() for m in re.finditer(rf"^{keyword}\s+(.*)$", joined, re.M)]


def json_list(value: str) -> list[str]:
    parsed = json.loads(value)
    assert isinstance(parsed, list)
    return [str(x) for x in parsed]


@pytest.mark.parametrize(
    ("name", "module"),
    [("engine", "engine.server"), ("router", "router.server"), ("bench", "bench.loadgen")],
)
def test_the_entrypoint_is_a_real_module(name: str, module: str) -> None:
    entry = json_list(instruction(dockerfile(name), "ENTRYPOINT")[0])
    assert entry == ["python", "-m", module]
    assert importlib.util.find_spec(module) is not None


def test_default_commands_parse() -> None:
    engine_cmd = json_list(instruction(dockerfile("engine"), "CMD")[0])
    args = parse_engine(engine_cmd)
    assert args.host == "0.0.0.0" and args.port == 8000
    router_cmd = json_list(instruction(dockerfile("router"), "CMD")[0])
    parsed = parse_router([*router_cmd, "--replica", "http://engine-0.engine:8000"])
    assert parsed.host == "0.0.0.0" and parsed.port == 9000
    with pytest.raises(SystemExit):  # the replicas really are required arguments
        parse_router(router_cmd)
    # the load generator's default command is --help, which prints the usage and exits cleanly
    with pytest.raises(SystemExit) as exit_info:
        bench.loadgen.build_parser().parse_args(
            json_list(instruction(dockerfile("bench"), "CMD")[0])
        )
    assert exit_info.value.code == 0


def test_the_healthchecks_and_ports_match_the_servers() -> None:
    for name, port in (("engine", 8000), ("router", 9000)):
        text = dockerfile(name)
        assert instruction(text, "EXPOSE") == [str(port)]
        check = instruction(text, "HEALTHCHECK")[0]
        assert f"http://127.0.0.1:{port}/health" in check


def test_the_images_run_as_the_numeric_user_kubernetes_expects() -> None:
    docs = base_docs()
    pod_specs = [
        docs["StatefulSet/engine"]["spec"]["template"]["spec"],
        docs["Deployment/router"]["spec"]["template"]["spec"],
    ]
    for name in ("engine", "router", "bench"):
        assert instruction(dockerfile(name), "USER") == ["10001:10001"]
    for spec in pod_specs:
        assert spec["securityContext"]["runAsNonRoot"] is True
        assert spec["securityContext"]["runAsUser"] == 10001
        assert spec["containers"][0]["securityContext"]["allowPrivilegeEscalation"] is False


def test_the_images_install_only_what_their_program_needs() -> None:
    engine, router_text, bench_text = (
        dockerfile("engine"),
        dockerfile("router"),
        dockerfile("bench"),
    )
    assert "torch" in " ".join(instruction(engine, "RUN"))
    assert "torch" not in " ".join(instruction(router_text, "RUN"))
    assert "torch" not in " ".join(instruction(bench_text, "RUN"))
    assert instruction(engine, "COPY") == [
        "constraints.txt /app/constraints.txt",
        "engine /app/engine",
    ]
    assert "gcc" in " ".join(instruction(engine, "RUN"))  # Triton compiles a launcher


def test_the_router_really_does_not_import_torch() -> None:
    code = "import sys, router.server; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_the_dockerignore_keeps_secrets_weights_and_results_out() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for needed in ("models", "results", ".git", ".env", "*.safetensors", "*.pem", "hf_token*"):
        assert needed in ignored


# -- docker compose ------------------------------------------------------------------------------


def test_compose_commands_parse_and_the_router_names_the_engine_services() -> None:
    compose = yaml.safe_load((DOCKER / "compose.mock.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in ("engine-0", "engine-1"):
        args = parse_engine(services[name]["command"])
        assert args.backend == "mock" and args.host == "0.0.0.0"
    router_args = parse_router(services["router"]["command"])
    assert router_args.replica == [f"http://engine-{n}:8000" for n in range(2)]
    assert set(services["router"]["depends_on"]) == {"engine-0", "engine-1"}
    load = bench.loadgen.build_parser().parse_args(services["loadgen"]["command"])
    assert load.url == "http://router:9000" and load.command == "run"
    assert services["loadgen"]["profiles"] == ["load"]
    # the images the compose file builds carry the tag the Kubernetes files use
    for service in ("engine-0", "router", "loadgen"):
        assert services[service]["image"].endswith(f":{IMAGE_TAG}")
    for service, dockerfile_name in (
        ("engine-0", "engine"),
        ("router", "router"),
        ("loadgen", "bench"),
    ):
        assert (
            services[service]["build"]["dockerfile"]
            == f"deploy/docker/Dockerfile.{dockerfile_name}"
        )


def test_the_single_gpu_overlay_has_one_replica_and_the_router_agrees() -> None:
    statefulset = patched("single-gpu", "StatefulSet", "engine")
    router = patched("single-gpu", "Deployment", "router")
    assert statefulset["spec"]["replicas"] == 1
    engine_args = parse_engine(engine_container(statefulset)["args"])
    router_args = parse_router(router["spec"]["template"]["spec"]["containers"][0]["args"])
    assert router_args.replica == [f"http://engine-0.engine:{engine_args.port}"]
    # everything else about the router is as in the base
    base_router = parse_router(
        base_docs()["Deployment/router"]["spec"]["template"]["spec"]["containers"][0]["args"]
    )
    assert (router_args.policy, router_args.load_cap, router_args.port) == (
        base_router.policy,
        base_router.load_cap,
        base_router.port,
    )
    # and it still asks for exactly one GPU, as the base does
    assert engine_container(statefulset)["resources"]["limits"]["nvidia.com/gpu"] == 1


def test_the_mock_overlay_uses_the_cpu_build_of_the_engine_image() -> None:
    kustomization = yaml.safe_load((K8S / "overlays" / "mock" / "kustomization.yaml").read_text())
    tags = {i["name"]: i["newTag"] for i in kustomization["images"]}
    assert tags == {"pagedserve-engine": f"{IMAGE_TAG}-cpu"}
    # the Dockerfile really offers that build: the index is an argument that can be set to "cpu"
    assert "ARG TORCH_CUDA_INDEX=cu132" in dockerfile("engine")
    assert 'whl/${TORCH_CUDA_INDEX}"' in dockerfile("engine")


def test_every_image_and_the_compose_file_carry_the_commit() -> None:
    """Results made in a container must be able to say which code ran (no .git inside an image)."""
    for name in ("engine", "router", "bench"):
        text = dockerfile(name)
        assert instruction(text, "ARG").count("PAGEDSERVE_COMMIT=unknown") == 1
        assert "PAGEDSERVE_COMMIT=${PAGEDSERVE_COMMIT}" in instruction(text, "ENV")
        # late in the file, after the code is copied, so that a new commit keeps the cached layers
        lines = text.splitlines()
        arg_line = next(i for i, ln in enumerate(lines) if ln.startswith("ARG PAGEDSERVE_COMMIT"))
        copy_line = max(i for i, ln in enumerate(lines) if ln.startswith("COPY "))
        assert arg_line > copy_line
    compose = yaml.safe_load((DOCKER / "compose.mock.yaml").read_text(encoding="utf-8"))
    for service in ("engine-0", "router", "loadgen"):
        args = compose["services"][service]["build"]["args"]
        assert args["PAGEDSERVE_COMMIT"] == "${PAGEDSERVE_COMMIT:-unknown}"

# Deploying pagedserve: Docker and Kubernetes

**Status (2026-11-24): written and checked offline, never built or applied.** No Docker, cluster
or GPU was available while these files were written. What has been verified, and what has not, is
spelled out at the end. Treat the first build and the first `kubectl apply` as the first real test,
and send what breaks back into this document.

| Path | What it is |
| --- | --- |
| `docker/Dockerfile.engine` | the engine server (GPU; PyTorch for CUDA 13.2 with the benchmark pins) |
| `docker/Dockerfile.router` | the router (no GPU, no PyTorch) |
| `docker/Dockerfile.bench` | the load generator |
| `docker/compose.mock.yaml` | a whole stack on one machine with no GPU and no weights (mock backend) |
| `k8s/base/` | namespace, models volume claim, engine StatefulSet and headless Service, router Deployment and Service |
| `k8s/overlays/kernel-graphs/` | the engine with the opt-in Triton kernel and CUDA graphs |
| `k8s/overlays/mock/` | the stack on a cluster with no GPU and no weights (mock backend) |
| `validate.sh`, `validate.ps1` | the offline checks (below) |
| `setup_gpu_box.sh`, `capsule/`, `gpu_*.sh` | the earlier, non-container way the benchmarks were run (a shared GPU machine) |

## Prerequisites

- **Images:** Docker (or another OCI builder). The engine image downloads PyTorch with CUDA (a few
  gigabytes), even for the mock demo, because `engine.server` imports it.
- **GPU in containers:** an NVIDIA driver on the host and the NVIDIA Container Toolkit. The CUDA
  runtime is inside the PyTorch wheels, so the image itself does not need a CUDA base image. The
  benchmark machine's driver advertised CUDA 13.0 while the wheel is built for 13.2 and still passed
  the determinism self-test (see `setup_gpu_box.sh` and D-017); on a new machine run
  `python -m bench.gpu_selftest` inside the container before trusting any number.
- **Kubernetes:** a cluster with the NVIDIA device plugin (so `nvidia.com/gpu` is schedulable) and a
  storage class that lets several nodes read one volume (the weights volume is shared).
- **Weights:** Llama-3.2-1B-Instruct is licence-gated. Accept Meta's licence on Hugging Face, download
  the six files yourself, and never put them or a token in an image or in this repository.

## 1. Try the whole stack without a GPU (Docker)

```bash
docker compose -f deploy/docker/compose.mock.yaml up --build
curl http://localhost:9000/info                       # the router and, with ?fleet=true, each replica
docker compose -f deploy/docker/compose.mock.yaml --profile load run --rm loadgen
```

The mock backend has placeholder timings (`engine/mock.py`): this exercises routing, streaming and
health checks, and says nothing about performance. Results of the load run land in `deploy/docker/out/`.

## 2. Build the images

From the repository root:

```bash
docker build -f deploy/docker/Dockerfile.engine -t pagedserve-engine:0.1.0 .
docker build -f deploy/docker/Dockerfile.router -t pagedserve-router:0.1.0 .
docker build -f deploy/docker/Dockerfile.bench  -t pagedserve-bench:0.1.0 .
```

Push them to a registry the cluster can read and set `images:` in `k8s/base/kustomization.yaml` (or
`kustomize edit set image`) to the registry names and tag. A single engine container on one GPU:

```bash
docker run --rm --gpus all -p 8000:8000 -v /path/to/models:/models:ro pagedserve-engine:0.1.0
# the same, with the opt-in speed paths (read their entries in docs/decisions.md first):
docker run --rm --gpus all -p 8000:8000 -v /path/to/models:/models:ro pagedserve-engine:0.1.0 \
    --model-dir /models/Llama-3.2-1B-Instruct --host 0.0.0.0 --paged-attention-kernel --cuda-graphs
```

## 3. Kubernetes

Without a GPU (a local `kind` or `minikube` cluster is enough):

```bash
kubectl apply -k deploy/k8s/overlays/mock
kubectl -n pagedserve rollout status statefulset/engine deployment/router
kubectl -n pagedserve port-forward svc/router 9000:80 &
curl http://localhost:9000/info
```

On a GPU cluster:

```bash
kubectl apply -k deploy/k8s/base               # or overlays/kernel-graphs
# put the weights on the volume: a throwaway pod that mounts the claim, then copy the files in
kubectl -n pagedserve run models-loader --image=busybox --restart=Never \
    --overrides='{"spec":{"containers":[{"name":"l","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"m","mountPath":"/models"}]}],"volumes":[{"name":"m","persistentVolumeClaim":{"claimName":"pagedserve-models"}}]}}'
kubectl -n pagedserve cp /path/to/Llama-3.2-1B-Instruct models-loader:/models/Llama-3.2-1B-Instruct
kubectl -n pagedserve delete pod models-loader
kubectl -n pagedserve rollout status statefulset/engine deployment/router
```

The engine pods keep restarting until the weights are in place; that is expected. The router replica
list (`--replica http://engine-N.engine:8000`) is fixed by the StatefulSet's pod names: if you change
`replicas` in `k8s/base/engine.yaml`, change the router's arguments too (a test fails if they disagree).

The router's `--load-cap 48` is a starting point only. D-007 found that this cap decides whether
prefix routing happens at all, and the router has never been run on real replicas (D-022).

## Checks that need no Docker or cluster

```bash
bash deploy/validate.sh          # Linux or macOS
.\deploy\validate.ps1            # Windows
```

They run hadolint on the Dockerfiles, render each kustomization with `kubectl kustomize` and check
it against the Kubernetes 1.34 schemas with kubeconform, and run `tests/test_deploy_files.py`, which:
parses every container, compose and manifest argument list with the servers' real argument parsers;
checks that the router's replica names, the Services' ports and selectors, the probe paths (against
the apps' real routes) and the image tags agree across files; applies the overlays' patches to check
the mock overlay really has no GPU and no weights volume and the kernel overlay only adds its two
flags; and checks that the router really does not import PyTorch and that the images run as the
numeric user the pods expect. Every one of those checks was shown to fail when the file it guards
was deliberately broken.

## What is verified and what is not

Verified (offline, 2026-11-24): the Dockerfiles pass hadolint; the three kustomizations render and
validate against the Kubernetes 1.34 schemas; the argument lists parse; names, ports, routes and tags
agree; the overlays do what they claim.

**Not verified: that any image builds; that a container starts; that the GPU is visible inside it;
that Triton can compile its kernel there; that the probes, the read-only root file system and the
non-root user work with PyTorch and Triton; that the pods schedule; that the router reaches the
replicas.** If a first start fails, try these in this order:
1. `readOnlyRootFilesystem` (set it to `false` in `k8s/base/*.yaml` to see whether a library wants to
   write somewhere other than `/tmp` and `/home/app/.triton`);
2. the Triton compile (it needs `gcc`, installed in the engine image) and the driver against the
   CUDA 13.2 wheel (run the self-test in the container);
3. the weights volume (path `/models/Llama-3.2-1B-Instruct`, mounted read-only, readable by uid 10001);
4. the startup probe budget (10 minutes: weights load, kernel compile and graph capture happen
   before the server answers).

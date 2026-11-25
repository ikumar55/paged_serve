# Deploying pagedserve: Docker and Kubernetes

**Status (2026-11-24): the three images build; the mock stack runs on Docker Desktop (WSL 2) and, through
the `mock` overlay, on a real local Kubernetes cluster (`kind`) on a Windows laptop. The GPU path has
never been run: no GPU was available.** What has been verified, and what has not, is spelled out at the
end. Treat the first GPU run as the first real test of that part, and send what breaks back into this
document.

| Path | What it is |
| --- | --- |
| `docker/Dockerfile.engine` | the engine server (GPU; PyTorch for CUDA 13.2 with the benchmark pins) |
| `docker/Dockerfile.router` | the router (no GPU, no PyTorch) |
| `docker/Dockerfile.bench` | the load generator |
| `docker/compose.mock.yaml` | a whole stack on one machine with no GPU and no weights (mock backend) |
| `k8s/base/` | namespace, models volume claim, engine StatefulSet and headless Service, router Deployment and Service |
| `k8s/overlays/kernel-graphs/` | the engine with the opt-in Triton kernel and CUDA graphs |
| `k8s/overlays/mock/` | the stack on a cluster with no GPU and no weights (mock backend, CPU build of the engine image) |
| `k8s/overlays/single-gpu/` | one engine replica and a router that knows only that one, for a machine with a single GPU |
| `validate.sh`, `validate.ps1` | the offline checks (below) |
| `setup_gpu_box.sh`, `capsule/`, `gpu_*.sh` | the earlier, non-container way the benchmarks were run (a shared GPU machine) |

## Prerequisites

- **Images:** Docker (or another OCI builder). The engine image is about 9 GB, because it carries
  PyTorch with CUDA, even for the mock demo (`engine.server` imports PyTorch); the router image is
  213 MB and the load generator image 416 MB.
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
health checks, and says nothing about performance (the load generator itself warns that a result
without `--server-env` must not be published). Results of the load run land in `deploy/docker/out/`,
which is git-ignored.

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

Without a GPU, on a local `kind` cluster (this is what was tested). The mock overlay uses the CPU
build of the engine image (about 1.8 GB instead of 9 GB):

```bash
docker build --build-arg TORCH_CUDA_INDEX=cpu -f deploy/docker/Dockerfile.engine -t pagedserve-engine:0.1.0-cpu .
docker build -f deploy/docker/Dockerfile.router -t pagedserve-router:0.1.0 .
kind create cluster --name pagedserve
kind load docker-image pagedserve-engine:0.1.0-cpu pagedserve-router:0.1.0 --name pagedserve
kubectl apply -k deploy/k8s/overlays/mock
kubectl -n pagedserve rollout status statefulset/engine deployment/router
kubectl -n pagedserve port-forward svc/router 9000:80 &
curl "http://localhost:9000/info?fleet=true"
kind delete cluster --name pagedserve        # when done
```

On a GPU cluster (`overlays/single-gpu` for a machine with one GPU: it runs one engine replica,
because a second pod asking for its own GPU would stay Pending, and the router lists only that one):

```bash
kubectl apply -k deploy/k8s/base               # or overlays/single-gpu, overlays/kernel-graphs
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

Verified offline (2026-11-24): the Dockerfiles pass hadolint; the three kustomizations render and
validate against the Kubernetes 1.34 schemas; the argument lists parse; names, ports, routes and tags
agree; the overlays do what they claim.

**Verified by running it on a real Kubernetes cluster (2026-11-24, `kind` 0.33, Kubernetes 1.37, on
the same laptop, mock overlay):** `kubectl apply -k deploy/k8s/overlays/mock` brought up `engine-0`,
`engine-1` and the router, all Ready, on the first apply; the router found both replicas through the
StatefulSet's DNS names (`engine-0.engine:8000`) and a request through the Service returned tokens; the
load generator container ran 160 requests through the router, 160 of 160 succeeded and both replicas
served; the pod security settings are really in force (user 10001, a write to `/` refused with "Read-only
file system", `/tmp` writable); deleting `engine-0` during traffic caused no failed request (45 of 45), one
router failover, and the StatefulSet recreated the pod with the same name; and applying the
`single-gpu` overlay on this GPU-less cluster leaves `engine-0` Pending with `Insufficient
nvidia.com/gpu` (and `Insufficient memory`, because the base asks for 8 GiB and the laptop's Docker VM has
less), which shows the GPU and memory requests are enforced by the scheduler.

**Verified by running it with Docker (2026-11-24, Docker Desktop 4.100 with WSL 2, Docker 29.9, a Windows
laptop with no GPU):** all three images build from the Dockerfiles on the first attempt (the engine image in
about 4 minutes, mostly the PyTorch download); `docker compose -f deploy/docker/compose.mock.yaml up`
starts two mock engine replicas and the router, all three report healthy through the images' own
`HEALTHCHECK`s within seconds; the router lists both replicas healthy and a request through it returns
tokens; the load generator container ran 160 requests through the router, 160 of 160 succeeded, and wrote its
result file; stopping one engine container leaves requests succeeding (8 of 8, one failover, the
other replica takes the load) and the router takes the replica back after it restarts. Run as the
non-root user 10001 throughout.

**Not verified: that the GPU is visible inside a container; that Triton can compile its kernel
there; that the base and `kernel-graphs` overlays (which need a GPU and the weights) start on a GPU
cluster; that the weights volume mounts and is readable by user 10001; scheduling of two GPU replicas
onto two GPUs.** If a first GPU start fails, try these in this order:
1. `readOnlyRootFilesystem` (set it to `false` in `k8s/base/*.yaml` to see whether a library wants to
   write somewhere other than `/tmp` and `/home/app/.triton`);
2. the Triton compile (it needs `gcc`, installed in the engine image) and the driver against the
   CUDA 13.2 wheel (run the self-test in the container);
3. the weights volume (path `/models/Llama-3.2-1B-Instruct`, mounted read-only, readable by uid 10001);
4. the startup probe budget (10 minutes: weights load, kernel compile and graph capture happen
   before the server answers).

## Found while testing, and fixed

- **The router's `/health` was optimistic.** A replica counted as healthy until a *request* to it
  failed, so with the only replica missing the router's `/health` returned 200 and Kubernetes marked it
  Ready; the first request then returned 503 (found on `kind` with the `single-gpu` overlay while the
  engine pod was Pending). Fixed (2026-11-24): until some replica has answered once (a probe of its
  `/health`, a successful request or a successful `/cache` poll), the router's `/health` asks the
  replicas and returns 503 `no replica has answered yet` if none answers 200. After the first contact it
  goes back to the cooldown flags as before, deliberately: a busy replica that is slow to answer must not
  make the router unready, and a router that lost every replica after having served already reports 503
  through the cooldown. **Re-run on the `kind` cluster:** with no backend (`single-gpu` overlay on a
  GPU-less cluster) the router is `0/1` not Ready, where before the fix it was `1/1`; when the engines can
  start (`mock` overlay) all three pods are Ready within 15 seconds. Covered by six tests, with every
  deliberate breakage caught.
- **The router's liveness probe is a TCP check, not `/health`.** With `/health` as the liveness probe,
  Kubernetes would restart the router for being not ready while the engines were still loading their
  weights (minutes on a GPU). Alive means the process accepts connections; ready means a backend
  answers. Checked on the cluster: 170 seconds with no backend, router not Ready, zero restarts.

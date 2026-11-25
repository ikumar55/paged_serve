#!/usr/bin/env bash
# Offline checks of the deployment files: no Docker, no cluster, no GPU needed.
#
#   bash deploy/validate.sh
#
# 1. hadolint on every Dockerfile (skipped with a notice if hadolint is not installed);
# 2. `kubectl kustomize` renders the base and each overlay, and kubeconform checks the output
#    against the Kubernetes schemas (skipped with a notice if either tool is missing);
# 3. the tests that parse every argument list with the servers' real parsers and check that names,
#    ports and routes agree across files (needs the project's Python environment).
#
# What this does NOT prove: that an image builds, that a pod starts, that the GPU is visible.
# Those need Docker and a cluster (see deploy/README.md).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
KUBE_VERSION="${KUBE_VERSION:-1.34.0}"
PYTHON="${PYTHON:-python}"
status=0

step() { printf '\n== %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

step "hadolint"
if have hadolint; then
  hadolint deploy/docker/Dockerfile.engine deploy/docker/Dockerfile.router deploy/docker/Dockerfile.bench || status=1
else
  echo "hadolint not installed: skipped"
fi

step "kubectl kustomize + kubeconform (Kubernetes $KUBE_VERSION schemas)"
if have kubectl && have kubeconform; then
  for dir in deploy/k8s/base deploy/k8s/overlays/mock deploy/k8s/overlays/kernel-graphs \
             deploy/k8s/overlays/single-gpu; do
    printf '%s: ' "$dir"
    kubectl kustomize "$dir" | kubeconform -strict -summary -kubernetes-version "$KUBE_VERSION" - || status=1
  done
else
  echo "kubectl or kubeconform not installed: skipped"
fi

step "deployment tests"
"$PYTHON" -m pytest tests/test_deploy_files.py -q || status=1

exit "$status"

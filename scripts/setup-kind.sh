#!/bin/bash
set -euo pipefail

CLUSTER_NAME="docs-agent"
NAMESPACE="docs-agent"

echo "=== Kubeflow Docs Agent - Kind Deployment ==="

# create cluster if it doesn't exist
if kind get clusters 2>/dev/null | grep -q "$CLUSTER_NAME"; then
  echo "Cluster '$CLUSTER_NAME' already exists, reusing"
else
  echo "Creating Kind cluster '$CLUSTER_NAME'..."
  kind create cluster --name "$CLUSTER_NAME" --wait 60s
fi

kubectl config use-context "kind-$CLUSTER_NAME"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# build container images
echo "Building container images..."
docker build -t docs-agent-ingestion:latest -f ingestion/Dockerfile .
docker build -t docs-agent-server:latest -f server/Dockerfile .
docker build -t docs-agent-frontend:latest ./frontend

# load images into Kind (avoids needing a registry)
echo "Loading images into Kind cluster..."
kind load docker-image docs-agent-ingestion:latest --name "$CLUSTER_NAME"
kind load docker-image docs-agent-server:latest --name "$CLUSTER_NAME"
kind load docker-image docs-agent-frontend:latest --name "$CLUSTER_NAME"

# deploy via kustomize
echo "Deploying application manifests..."
kubectl apply -k k8s/

# wait for Milvus
echo "Waiting for Milvus to be ready..."
kubectl wait --for=condition=ready pod -l app=milvus -n "$NAMESPACE" --timeout=180s || true

# wait for server
echo "Waiting for server to be ready..."
kubectl wait --for=condition=ready pod -l app=docs-agent-server -n "$NAMESPACE" --timeout=120s || true

# run ingestion as a one-shot job
echo "Running ingestion pipeline..."
kubectl run docs-agent-ingest \
  --namespace="$NAMESPACE" \
  --image=docs-agent-ingestion:latest \
  --restart=Never \
  --env="MILVUS_URI=http://milvus.$NAMESPACE.svc.cluster.local:19530" \
  --env="GITHUB_TOKEN=${GITHUB_TOKEN:-}" \
  --rm -it --quiet 2>/dev/null || echo "Ingestion may need to be run manually"

# port-forward for local access
echo ""
echo "=== Deployment Complete ==="
echo "Starting port-forwards (Ctrl+C to stop)..."
echo ""
echo "  Frontend: http://localhost:8080"
echo "  API:      http://localhost:8000"
echo "  MCP:      http://localhost:8001"
echo "  Health:   curl http://localhost:8000/health"
echo ""

kubectl port-forward svc/docs-agent-frontend 8080:80 -n "$NAMESPACE" &
kubectl port-forward svc/docs-agent-server 8000:8000 -n "$NAMESPACE" &
kubectl port-forward svc/docs-agent-server 8001:8001 -n "$NAMESPACE" &

wait

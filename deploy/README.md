# Deployment

Infrastructure for running QuantMind beyond a local process. These manifests are
**deploy-ready but require your own cluster / cloud account** to actually apply —
nothing here runs without one.

## Kubernetes (`k8s/`)

```bash
# Build & push the image first (set OWNER in deployment.yaml to your registry).
docker build -t ghcr.io/OWNER/quantmind:latest .
docker push ghcr.io/OWNER/quantmind:latest

kubectl apply -f deploy/k8s/
kubectl get svc quantmind-api
```

- `deployment.yaml` — 2 replicas, health/readiness probes on `/health`, resource
  limits, optional `ANTHROPIC_API_KEY` secret.
- `service.yaml` — ClusterIP Service + ConfigMap.
- `hpa.yaml` — HorizontalPodAutoscaler (CPU 70%, 2–8 replicas) + nginx Ingress.

## Terraform — AWS ECS Fargate (`terraform/`)

```bash
cd deploy/terraform
terraform init
terraform apply -var="image=<your-ecr-image-uri>"
# -> outputs alb_dns_name (the app URL) and ecr_repository_url
```

Provisions an ECR repo, ECS Fargate cluster + service (2 tasks), an Application
Load Balancer with a `/health` health check, security groups, IAM execution
role, and CloudWatch logs — using the account's default VPC.

## Local full stack (no cloud)

`docker compose up --build` from the repo root brings up the API + Qdrant +
Kafka locally. See the root `docker-compose.yml`.

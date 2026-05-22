output "alb_dns_name" {
  description = "Public DNS name of the load balancer (the app URL)."
  value       = aws_lb.this.dns_name
}

output "ecr_repository_url" {
  description = "Push your container image here, then pass it as -var=image=..."
  value       = aws_ecr_repository.this.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

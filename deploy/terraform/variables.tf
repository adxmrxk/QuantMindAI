variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Resource name prefix."
  type        = string
  default     = "quantmind"
}

variable "image" {
  description = "Container image URI (e.g. an ECR image pushed from the Dockerfile)."
  type        = string
}

variable "container_port" {
  description = "Port the API listens on."
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "Fargate task CPU units (1024 = 1 vCPU)."
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate task memory (MiB)."
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of running tasks."
  type        = number
  default     = 2
}

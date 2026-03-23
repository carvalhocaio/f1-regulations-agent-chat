variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "f1-regulations-agent-chat"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (staging or production)"
  type        = string
}

variable "sa_name" {
  description = "Service account name"
  type        = string
  default     = "f1-agent-engine"
}

variable "staging_bucket_name" {
  description = "GCS bucket for staging artifacts"
  type        = string
  default     = "f1-agent-staging"
}

variable "agent_display_name" {
  description = "Display name for Agent Engine"
  type        = string
}

variable "agent_description" {
  description = "Description for Agent Engine"
  type        = string
  default     = "AI assistant for Formula 1 regulations and history"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 5
}

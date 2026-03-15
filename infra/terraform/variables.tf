variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-1"
}

variable "az" {
  description = "Availability Zone for the public subnet"
  type        = string
  default     = "ap-northeast-1a"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH into instances (e.g., your laptop /32)"
  type        = string
}

variable "extra_tags" {
  description = "Additional tags to merge"
  type        = map(string)
  default     = {}
}

variable "instance_state" {
  type    = string
  default = "running"

  validation {
    condition     = contains(["running", "stopped"], var.instance_state)
    error_message = "instance_state must be running or stopped"
  }
}

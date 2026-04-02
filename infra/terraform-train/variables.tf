variable "aws_region" {
  default = "us-east-1"
}

variable "instance_type" {
  default     = "g4dn.xlarge"
  description = "GPU spot instance type. g4dn.xlarge = 1x T4 GPU, 4 vCPU, 16 GB RAM. ~$0.16/hr spot."
}

variable "spot_max_price" {
  default     = "0.50"
  description = "Max spot bid price in USD/hr. On-demand for g4dn.xlarge is ~$0.526."
}

variable "key_pair_name" {
  type = string
}

variable "your_ip_cidr" {
  type = string
}

variable "project_name" {
  default = "covered-call-train"
}

variable "repo_url" {
  type        = string
  description = "HTTPS URL of your GitHub repo"
}

variable "aws_region" {
  default = "us-east-1"
}

variable "instance_type" {
  default = "t3.medium"   # 2 vCPU, 4 GB — sufficient for CPU inference
}

variable "key_pair_name" {
  type = string
}

variable "your_ip_cidr" {
  type = string
}

variable "project_name" {
  default = "covered-call-app"
}

variable "repo_url" {
  type        = string
  description = "HTTPS URL of your GitHub repo, e.g. https://github.com/YOUR_USERNAME/capstone-aai-590.git"
}

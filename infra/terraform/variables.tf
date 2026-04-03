variable "aws_region" {
  description = "AWS region to deploy the training instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 GPU instance type. g4dn.xlarge = T4 GPU (cheapest). g4dn.2xlarge = T4 + more RAM."
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot" {
  description = "Use Spot Instance for ~70% cost savings. Set false for uninterruptible training."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Max hourly price for Spot Instance (on-demand price of g4dn.xlarge is ~$0.526)"
  type        = string
  default     = "0.30"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "your_ip_cidr" {
  description = "Your public IP in CIDR notation for SSH access. Find it at https://checkip.amazonaws.com"
  type        = string
  # Example: "203.0.113.45/32"
}

variable "s3_data_bucket" {
  description = "S3 bucket name where your parquet data file is stored"
  type        = string
}

variable "s3_data_key" {
  description = "S3 key (path) to the parquet data file"
  type        = string
  default     = "data/clean/daily_stock_optimal_bucket_modeling_with_fred.parquet"
}

variable "volume_size_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 60
}

variable "project_name" {
  description = "Tag prefix applied to all resources"
  type        = string
  default     = "covered-call-ml"
}

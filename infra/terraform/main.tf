terraform {
  required_version = ">= 1.3"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Data sources ────────────────────────────────────────────────────────────

# AWS Deep Learning AMI (GPU PyTorch) — comes with CUDA, cuDNN, PyTorch pre-installed
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── IAM role — allows EC2 to read/write S3 ──────────────────────────────────

resource "aws_iam_role" "training" {
  name = "${var.project_name}-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = { Project = var.project_name }
}

resource "aws_iam_role_policy" "s3_access" {
  name = "${var.project_name}-s3-access"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_data_bucket}",
          "arn:aws:s3:::${var.s3_data_bucket}/*",
          # Also allow writing trained models back to S3
          "arn:aws:s3:::${var.s3_data_bucket}/saved_models/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "${var.project_name}-instance-profile"
  role = aws_iam_role.training.name
}

# ── Security group — SSH only from your IP ──────────────────────────────────

resource "aws_security_group" "training" {
  name        = "${var.project_name}-sg"
  description = "SSH access for ML training instance"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from your IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  # Jupyter Lab (optional — useful for browser-based notebook access)
  ingress {
    description = "JupyterLab"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-sg", Project = var.project_name }
}

# ── EC2 instance (On-Demand or Spot) ────────────────────────────────────────

resource "aws_instance" "training" {
  count = var.use_spot ? 0 : 1

  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.training.id]
  iam_instance_profile   = aws_iam_instance_profile.training.name

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_data_bucket = var.s3_data_bucket
    s3_data_key    = var.s3_data_key
  }))

  tags = {
    Name    = "${var.project_name}-training"
    Project = var.project_name
  }
}

# Spot instance request (cheaper, ~70% savings)
resource "aws_spot_instance_request" "training" {
  count = var.use_spot ? 1 : 0

  ami                            = data.aws_ami.dlami.id
  instance_type                  = var.instance_type
  key_name                       = var.key_pair_name
  subnet_id                      = data.aws_subnets.default.ids[0]
  vpc_security_group_ids         = [aws_security_group.training.id]
  iam_instance_profile           = aws_iam_instance_profile.training.name
  spot_price                     = var.spot_max_price
  wait_for_fulfillment           = true
  instance_interruption_behavior = "stop"

  root_block_device {
    volume_size           = var.volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    s3_data_bucket = var.s3_data_bucket
    s3_data_key    = var.s3_data_key
  }))

  tags = {
    Name    = "${var.project_name}-training-spot"
    Project = var.project_name
  }
}

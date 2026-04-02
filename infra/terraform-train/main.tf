terraform {
  required_version = ">= 1.3"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.aws_region }

data "aws_vpc" "default" { default = true }

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ── Latest Deep Learning AMI (GPU) ──────────────────────────────────────────
# Uses Amazon's DL AMI which has CUDA, PyTorch, conda pre-installed.
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Amazon Linux 2023*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ── IAM role ────────────────────────────────────────────────────────────────
resource "aws_iam_role" "train" {
  name = "${var.project_name}-train-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3" {
  role       = aws_iam_role.train.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_instance_profile" "train" {
  name = "${var.project_name}-train-profile"
  role = aws_iam_role.train.name
}

# ── Security group ────────────────────────────────────────────────────────
resource "aws_security_group" "train" {
  name   = "${var.project_name}-train-sg"
  vpc_id = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  ingress {
    description = "Jupyter"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-train-sg" }
}

# ── Spot Instance Request ────────────────────────────────────────────────────
resource "aws_spot_instance_request" "train" {
  ami                            = data.aws_ami.dlami.id
  instance_type                  = var.instance_type
  spot_price                     = var.spot_max_price
  spot_type                      = "one-time"
  wait_for_fulfillment           = true
  instance_interruption_behavior = "terminate"

  key_name               = var.key_pair_name
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.train.id]
  iam_instance_profile   = aws_iam_instance_profile.train.name

  root_block_device {
    volume_size           = 100    # larger: DLAMI + datasets + model checkpoints
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/train_bootstrap.sh", {
    repo_url    = var.repo_url
    mlflow_port = 5000
  }))

  tags = { Name = "${var.project_name}-train-spot", Project = var.project_name }
}

locals {
  public_ip = var.use_spot ? (
    length(aws_spot_instance_request.training) > 0 ? aws_spot_instance_request.training[0].public_ip : ""
  ) : (
    length(aws_instance.training) > 0 ? aws_instance.training[0].public_ip : ""
  )
}

output "instance_public_ip" {
  description = "Public IP of the training instance"
  value       = local.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${local.public_ip}"
}

output "vscode_remote_host" {
  description = "Add this to ~/.ssh/config then connect via VSCode Remote SSH"
  value       = <<-EOT

    # Paste into ~/.ssh/config
    Host covered-call-gpu
        HostName ${local.public_ip}
        User ubuntu
        IdentityFile ~/.ssh/${var.key_pair_name}.pem
        ServerAliveInterval 60
        ServerAliveCountMax 3
  EOT
}

output "scp_data_command" {
  description = "Command to upload local data to instance (if not using S3)"
  value       = "scp -i ~/.ssh/${var.key_pair_name}.pem -r ./data/clean/ ubuntu@${local.public_ip}:~/covered-call-prototype/data/clean/"
}

output "jupyter_url" {
  description = "JupyterLab URL once the instance is ready (password: coveredcall)"
  value       = "http://${local.public_ip}:8888"
}

output "ami_used" {
  description = "Deep Learning AMI that was selected"
  value       = data.aws_ami.dlami.name
}

output "spot_instance_id"  { value = aws_spot_instance_request.train.spot_instance_id }
output "public_ip"         { value = aws_spot_instance_request.train.public_ip }
output "ssh_command"       { value = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_spot_instance_request.train.public_ip}" }
output "jupyter_url"       { value = "http://${aws_spot_instance_request.train.public_ip}:8888" }
output "mlflow_url"        { value = "http://${aws_spot_instance_request.train.public_ip}:5000" }
output "bootstrap_log"     { value = "ssh in and run: tail -f /var/log/train_bootstrap.log" }

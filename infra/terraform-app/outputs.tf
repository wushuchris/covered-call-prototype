output "instance_public_ip" { value = aws_instance.app.public_ip }
output "streamlit_url"      { value = "http://${aws_instance.app.public_ip}:8501" }
output "api_url"            { value = "http://${aws_instance.app.public_ip}:8000" }
output "api_docs_url"       { value = "http://${aws_instance.app.public_ip}:8000/docs" }
output "mlflow_url"         { value = "http://${aws_instance.app.public_ip}:5000" }
output "ssh_command"        { value = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_instance.app.public_ip}" }

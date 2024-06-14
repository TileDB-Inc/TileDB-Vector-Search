# Used to run ann-benchmarks on an EC2 instance and download the results.
#
# To run:
# - pip install ".[benchmarks]"
# - Set up your AWS credentials locally. You can set them in `~/.aws/credentials` to be picked up automatically.
# - Fill in the following details. You can create these in the EC2 console.
#   1. key_name: Your EC2 key pair name.
#   2. key_path: The to your local private key file.
#     -  Make sure to `chmod 400 /path/to/key.pem` after download.
# - python src/benchmarks/ann-benchmarks.py

import logging
import os
import socket
import time

import boto3
import paramiko

# You must fill these in before running the script:
key_name = "key_name"
key_path = "/path/to/key.pem"

if not os.path.exists(key_path):
    raise FileNotFoundError(
        f"Key file not found at {key_path}. Please set the correct path before running."
    )

# You do not need to change these.
security_group_ids = ["sg-04258b401ce76d246"]
# 64 vCPU, 512 GiB, EBS-Only.
instance_type = "r6i.16xlarge"
# Amazon Linux 2023 AMI 2023.4.20240528.0 x86_64 HVM kernel-6.1 - 64 bit (x86) - uefi-preferred.
ami_id = "ami-09e647bf7a368e505"
username = "ec2-user"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(results_dir, exist_ok=True)
log_file_path = os.path.join(results_dir, "ann-benchmarks-logs.txt")
if os.path.exists(log_file_path):
    open(log_file_path, "w").close()
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Create an EC2 client
ec2 = boto3.client("ec2")


def terminate_instance(instance_id):
    logger.info(f"Terminating instance {instance_id}...")
    ec2.terminate_instances(InstanceIds=[instance_id])
    logger.info(f"Instance {instance_id} terminated.")


def check_ssh_ready(public_dns, key_filename):
    """Poll until SSH is ready"""
    timeout = 60 * 2
    logger.info(f"Will poll for {timeout} seconds until SSH is ready.")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            ssh.connect(
                public_dns, username=username, key_filename=key_filename, timeout=20
            )
            ssh.close()
            logger.info("SSH is ready.")
            return True
        except Exception as e:
            logger.error(f"Waiting for SSH: {e}")
            time.sleep(15)
    return False


def execute_commands(ssh, commands):
    """Execute a list of commands on the SSH connection and stream the output"""
    for command in commands:
        logger.info(f"Executing command: {command}")
        stdin, stdout, stderr = ssh.exec_command(command)

        # Stream stdout
        for line in iter(stdout.readline, ""):
            logger.info(line.strip())

        # Stream stderr
        for line in iter(stderr.readline, ""):
            logger.error(line.strip())


try:
    # Launch an EC2 instance
    logger.info("Launching EC2 instance...")
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=security_group_ids,
        MinCount=1,
        MaxCount=1,
    )
    instance_id = response["Instances"][0]["InstanceId"]
    logger.info(f"Launched EC2 instance with ID: {instance_id}")

    # Wait for the instance to be in a running state.
    logger.info("Waiting for instance to enter running state...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    # Get the public DNS name of the instance.
    instance_description = ec2.describe_instances(InstanceIds=[instance_id])
    public_dns = instance_description["Reservations"][0]["Instances"][0][
        "PublicDnsName"
    ]
    logger.info(f"Public DNS of the instance: {public_dns}")

    # Tag the instance.
    instance_name = f"vector-search-ann-benchmarks-{socket.gethostname()}"
    logger.info(f"Will name the instance: {instance_name}")
    ec2.create_tags(
        Resources=[instance_id],
        Tags=[
            {
                "Key": "Name",
                "Value": instance_name,
            },
        ],
    )

    # Wait for SSH to be ready
    if not check_ssh_ready(public_dns=public_dns, key_filename=key_path):
        raise RuntimeError("SSH did not become ready in time")

    # Connect to the instance using paramiko
    logger.info("Connecting to the instance via SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(public_dns, username=username, key_filename=key_path)
    logger.info("Connected to the instance.")

    # Initial setup commands
    initial_commands = [
        "sudo yum update -y",
        "sudo yum install git -y",
        "sudo yum install python3.9-pip -y",
        "sudo yum install docker -y",
        "sudo service docker start",
        "sudo usermod -a -G docker ec2-user",
        "groups",
    ]
    execute_commands(ssh, initial_commands)

    # Reconnect to the instance to refresh group membership.
    logger.info("Reconnecting to the instance to refresh group membership...")
    ssh.close()
    time.sleep(10)
    ssh.connect(public_dns, username=username, key_filename=key_path)
    logger.info("Reconnected to the instance.")

    # Run the benchmarks.
    post_reconnect_commands = [
        "groups",
        "git clone https://github.com/TileDB-Inc/ann-benchmarks.git",
        "cd ann-benchmarks && pip3 install -r requirements.txt",
        "cd ann-benchmarks && python3 install.py --algorithm tiledb",
        "cd ann-benchmarks && python3 run.py --dataset sift-128-euclidean --algorithm tiledb-ivf-flat --force --batch",
        "cd ann-benchmarks && sudo chmod -R 777 results/sift-128-euclidean/10/tiledb-ivf-flat-batch",
        "cd ann-benchmarks && python3 create_website.py",
    ]
    execute_commands(ssh, post_reconnect_commands)

    # Download the results.
    remote_paths = [
        "/home/ec2-user/ann-benchmarks/sift-128-euclidean_10_euclidean-batch.png",
        "/home/ec2-user/ann-benchmarks/sift-128-euclidean_10_euclidean-batch.html",
    ]
    sftp = ssh.open_sftp()
    for remote_path in remote_paths:
        local_filename = os.path.basename(remote_path)
        local_path = os.path.join(results_dir, local_filename)
        logger.info(f"Downloading {remote_path} to {local_path}...")
        sftp.get(remote_path, local_path)
        logger.info(f"File downloaded to {local_path}.")
    logger.info("File downloading complete, closing the SFTP connection.")
    sftp.close()

    logger.info("Benchmarking complete, closing the SSH connection.")
    ssh.close()

except Exception as e:
    logger.error(f"Error occurred: {e}")
    if "instance_id" in locals():
        logger.info(f"Will terminate instance {instance_id}.")
        terminate_instance(instance_id)

else:
    logger.info(f"Finished, will try to terminate instance {instance_id}.")
    if "instance_id" in locals():
        logger.info(f"Will terminate instance {instance_id}.")
        terminate_instance(instance_id)

logger.info("Done.")

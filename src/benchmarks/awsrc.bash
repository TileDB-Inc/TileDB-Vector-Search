#!/bin/bash

# Define the following variables for your use case
instance_id="i-01234567890abcdef"
volume_id="vol-abcdefghihklmnopq"
region="us-east-1"
instance_ip=$(aws ec2 describe-instances --instance-ids i-0daab006867136323 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)


# Define check_instance_status if it is not already defined
if [[ "$(type -t check_instance_status)" != "function" ]]; then
    check_instance_status() {
	local local_instance_ip=${1:-${instance_ip}}
	local ssh_port=22
	local local_max_tries=${max_nc_tries:-5}
	local sleep_time=${nc_tries_sleep:-5}
	local timeout=${nc_timeout:-2}

	local tries=0
	while [ $tries -lt $local_max_tries ]; do

	    if ! host "${local_instance_ip}" >/dev/null 2>&1; then
		echo "Invalid host name: $local_instance_ip"
		return 1  # Exit the function with a non-zero status
	    fi

	    nc -z -G ${timeout} $local_instance_ip $ssh_port >/dev/null 2>&1
	    if [ $? -ne 0 ]; then
		tries=$((tries + 1))
		if [ $tries -lt $local_max_tries ];
		then
		    echo "Instance is not ready for SSH connection yet."
		    sleep $sleep_time
		fi
	    else
		echo "Instance is ready for SSH connection."
		return 0
	    fi
	done

	echo "Instance is not ready after ${local_max_tries} tries."
	return 1
    }
fi

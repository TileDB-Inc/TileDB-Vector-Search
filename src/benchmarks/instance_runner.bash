#!/bin/bash

################################################################################
#
# Script for remotely running the `ivf_flat` benchmark scripts on different
# EC2 instances.
#
# The script will stop the currently running instance (if it is not of the
# desired type), start the new instance, and execute the benchmarking script
# on the remote host.
#
# You will need to have the aws cli tools, nc, and ssh installed on your
# local host.  The script assumes you can connect to the remote instance
# with the command `ssh ec2`.  You can change the command as necessary in
# this script, or establish an `ec2` host in your `.ssh/config`.
#
################################################################################

# git_branch="lums/tmp/benchmark"
git_branch="main"
ntrials=1

max_nc_tries=12
nc_tries_sleep=8
instance_ip=$(aws ec2 describe-instances --instance-ids ${instance_id} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

if [ -f ~/.bash_awsrc ]; then
    . ~/.bash_awsrc
fi

# Specify the instance you want to execute on
for instance_type in r6a.24xlarge c6a.16xlarge c6a.4xlarge c6a.2xlarge t3.xlarge;
do

    # How do we want to identify this set of runs
    benchname="1b-${instance_type}-10k-125MiB"

    # Name of the script to execute (see definition of the `command` variable below)
    bash_script="ivf_flat_full.bash"

    echo "Benchmark name is ${benchname}, running script ${bash_script}"

    echo "Preparing to run ${instance_type}"
    current_instance_type=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].InstanceType' --output text)
    state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)

    echo "First stopping ${current_instance_type}"

    if [[ ${state} == "running" && ${current_instance_type} == "${instance_type}" ]]; then
	echo "${current_instance_type} is already ${state}"
    else

	echo "${current_instance_type} is in state ${state}"
	aws ec2 --region ${region} stop-instances --instance-ids ${instance_id}
	sleep 1
        if nc_timeout=1 max_nc_tries=1 check_instance_status;
	then
	    ssh ec2 "sync;sync;sync;sudo shutdown -h now"
	fi
	sleep 1

	state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)

	# Assume instance *will* stop (eventually)
	while [ "$state" != "stopped" ]; do
	    state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	    echo "Instance ${current_instance_type} is ${state}"
	    sleep 1  # Delay for 1 second
	done

	echo "Instance is ${state}"

	# Change instance type
	change_msg=$(aws ec2 --region ${region} modify-instance-attribute --instance-id ${instance_id} --instance-type ${instance_type})
	sleep 1
	current_instance_type=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].InstanceType' --output text)
	if [ "${current_instance_type}" != ${instance_type} ];
	then
	    echo "Could not change to ${instance_type} because ${change_msg}.  Skipping ${instance_type}."
	    continue
	fi

	aws ec2 --region ${region} start-instances --instance-ids ${instance_id}

	state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	while [ "$state" != "running" ]; do
	    state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	    echo "Instance is ${state}"
	    sleep 1  # Delay for 1 second
	done

	echo "Instance is ${state}"
	sleep 30
    fi


    # Make sure remote instance is ready to accept logins
    nc_tries=0

    while true; do
	if nc -G 2 -zv "${instance_ip}" 22 >/dev/null 2>&1; then
            echo "EC2 instance is ready for remote logins."
            break
	fi

	nc_tries=$((nc_tries + 1))

	if [ "$nc_tries" -eq "$max_nc_tries" ]; then
            echo "Maximum number of tries reached. EC2 instance is not ready for remote logins."
            break
	fi

	echo "EC2 instance is not ready yet. Retrying in $nc_tries_sleep seconds..."
	sleep "$nc_tries_sleep"
    done

    for ((i=1; i<=2; i++))
    do
	# nuke from space, it's the only way to be sure
	# ssh ec2 killall -u lums
        ssh ec2 "kill \$(ps auxw | fgrep feature | awk '{ print \$2 }')"
	sleep 1
    done
    ssh ec2 ps auxw | fgrep feature

    command="bash TileDB-Vector-Search/src/benchmarks/${bash_script}"
    for ((i=1; i<=${ntrials}; i++))
    do
	logname="${benchname}-$(date +'%Y%m%d-%H%M%S').log"
	ssh ec2 ${command} | tee ${logname}
    done
done

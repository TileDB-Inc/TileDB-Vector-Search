#!/bin/bashA

instance_id="i-0daab006867136323"
region="us-east-1"
# git_branch="lums/tmp/benchmark"
git_branch="lums/tmp/gemm2.0"
ntrials=1

max_nc_tries=12
nc_tries_sleep=8
instance_ip=$(aws ec2 describe-instances --instance-ids i-0daab006867136323 --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

if [ -f ~/.bash_awsrc ]; then
    . ~/.bash_awsrc
fi

#    ssh ec2 "cd feature-vector-prototype ; git commit -am \"Pause for benchmark [skip ci]\" ; git checkout ${git_branch}"
#    ssh ec2 "cd feature-vector-prototype/src/cmake-build-release ; make -C libtiledbvectorsearch ivf_hack"

# for instance_type in c6a.4xlarge c6a.2xlarge;
# for instance_type in c6a.16xlarge c6a.2xlarge;
for instance_type in r6a.24xlarge c6a.16xlarge c6a.4xlarge c6a.2xlarge t3.xlarge t1.micro;
do

    benchname="1b-${instance_type}-10k-125MiB"
    bash_script="1b-c6a-16x-10k-125MiB.bash"

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
    # feature-vector-prototype/experimental/benchmarks/1b-c6a-16x-125MiB.bash
    # 1b-c6a-16x-125MiB-2023-0613-1419.log


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

    command="bash feature-vector-prototype/src/benchmarks/${bash_script}"
    for ((i=1; i<=${ntrials}; i++))
    do
	logname="${benchname}-$(date +'%Y%m%d-%H%M%S').log"
	ssh ec2 ${command} | tee ${logname}
    done
done

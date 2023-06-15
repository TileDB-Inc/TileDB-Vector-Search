
instance_id="i-0daab006867136323"
region="us-east-1"

for instance_type in c6a.4xlarge c6a.2xlarge;
do
    current_instance_type=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].InstanceType' --output text)
    state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)

    if [[ ${state} == "running" && ${current_instance_type} == "${instance_type}" ]]; then
	echo ${current_instance_type} is ${state}
    else

	aws ec2 --region ${region} stop-instances --instance-ids ${instance_id}
	aws ec2 --region ${region} stop-instances --instance-ids ${instance_id}
	ssh ec2 "sync;sync;sync;sudo shutdown -h now"
	ssh ec2 "sync;sync;sync;sudo shutdown -h now"
	aws ec2 --region ${region} stop-instances --instance-ids ${instance_id}

	state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	while [ "$state" != "stopped" ]; do
	    state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	    echo "Instance is ${state}"
	    sleep 1  # Delay for 1 second
	done

	echo "Instance is ${state}"

	aws ec2 --region ${region} modify-instance-attribute --instance-id ${instance_id} --instance-type ${instance_type}
	aws ec2 --region ${region} start-instances --instance-ids ${instance_id}

	state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	while [ "$state" != "running" ]; do
	    state=$(aws ec2 --region ${region} describe-instances --instance-ids ${instance_id} --query 'Reservations[].Instances[].State.Name' --output text)
	    echo "Instance is ${state}"
	    sleep 1  # Delay for 1 second
	done

	echo "Instance is ${state}"
    fi
    # feature-vector-prototype/experimental/benchmarks/1b-c6a-16x-125MiB.bash
    # 1b-c6a-16x-125MiB-2023-0613-1419.log

    benchname="1b-${instance_type}-125MiB"
    bash_script="1b-c6a-16x-125MiB.bash"
    command="bash feature-vector-prototype/experimental/benchmarks/${bash_script}"
    
    for ((i=1; i<=2; i++))
    do
	logname="${benchname}-$(date +'%Y%m%d-%H%M%S').log"
	ssh ec2 ${command} | tee ${logname}
    done
done

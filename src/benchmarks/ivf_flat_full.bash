#!/bin/bash

################################################################################
#
# Run the `ivf_flat` benchmark on the 1B dataset on a large combinaation of
# data source, block sizes, query sizes, and nprobe sizes.  Each combination
# of parameters is run five times.  The benchmark will generate a large
# amount of ouput, which should be saved to a log file.
#
# The file `log_postprocessing.py` in this subdirectory will read a set of
# log files and create a csv file with performance results, suitable for
# importing into your favorite spreadsheet program.
#
# To help with reproducibility, before we run the benchmark itself, we
# emit some information about the execution environment (which is assumed
# to be an EC2 instance).
#
# It is assumed that the `ivf_flat` executable that you wish to run has
# been built and is pointed to by the appropriate variable in `setup.bash`
#
################################################################################

dir=$(dirname $0)

. ${dir}/setup.bash

printf "=========================================================================================================================================\n\n"
echo "Starting benchmark run: "
date +"%A, %B %d, %Y %H:%M:%S"
echo Running script $0
printf "Benchmark program ${ivf_query}\n\n"
uptime

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

#if ping -c 1 -W 1250 169.254.169.254;
if [[ -d "/sys/hypervisor/uuid" ]]
then
  echo "Running on EC2 instance"
  curl -s http://169.254.169.254/latest/meta-data/instance-type
  aws ec2 --region us-east-1 describe-volumes --volume-id ${volume_id}
else
  echo "Not running on EC2 instance"
  hostname
fi

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

arch
nproc
head -1 /proc/meminfo

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

declare -f ivf_query

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

cat $0

echo "========================================================================================================================================="

# Benchmark both local storage and cloud storage
for source in gp3 s3;
do
    init_1B_${source}
    for blocksize in 0 1000000 10000000 ;
    do
	for nqueries in 1 10 100 1000 10000;
	do
	    for nprobe in 1 2 4 8 16 32 64 128 ;
	    do
		ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize ${blocksize} --log -
		ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize ${blocksize} --log -
		ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize ${blocksize} --log -
		ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize ${blocksize} --log -
		ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize ${blocksize} --log -
	    done
	done
    done
done

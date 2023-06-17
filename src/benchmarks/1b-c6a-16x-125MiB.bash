#!/bin/bash

dir=$(dirname $0)

. ${dir}/setup.bash

ivf_query=~/TileDB/feature-vector-prototype/experimental/cmake-build-release/src/ivf_hack
ivf_query=/home/lums/feature-vector-prototype/experimental/cmake-build-release/libtiledbvectorsearch/src/ivf_hack

printf "=========================================================================================================================================\n\n"
echo "Starting benchmark run: "
date +"%A, %B %d, %Y %H:%M:%S"
echo $0
printf "Benchmark program ${ivf_query}\n\n"
uptime

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

curl -s http://169.254.169.254/latest/meta-data/instance-type

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

aws ec2 --region us-east-1 describe-volumes --volume-id vol-0192769447c7688d0 

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

arch
nproc
head -1 /proc/meminfo

printf "\n\n-----------------------------------------------------------------------------------------------------------------------------------------\n\n"

cat $0

echo "========================================================================================================================================="

for source in gp3 s3;
do
    init_1B_${source}
    for blocksize in 0 1000000 10000000 ;
    do
	log_header
	for nqueries in 1 10 100 ;
	do
	    for nprobe in 1 2 4 8 16 32 64 128 ;
	    do
		ivf_query --nqueries ${nqueries} --nprobe ${nprobe} --finite --blocksize ${blocksize}
	    done
	done
    done
done

#!/bin/bash
PATH=/usr/bin:$PATH

# This line may need to be changed. I'm not entirely sure to what, though...
DOCKER_HOST=unix:///run/user/10013/docker.sock
CONTAINER=suspension_of_disbelief
docker kill --signal=SIGINT disbelief
docker stop disbelief && docker rm disbelief

#`python EntitySubstituteTest.py <Task>   <Set> <DataDir> <Model Name>     <Blanks> <Passes> <Result Folder> <Batch Size> <Start At>`
#`python EntitySubstituteTest.py docred   dev   data      bert-large-cased 2        3        res             1000         0`
# num_blanks = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# num_passes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# data_path = sys.argv[3] if len(sys.argv) > 3 else "data"
# task_name = sys.argv[4] if len(sys.argv) > 4 else "docred"
# data_set = sys.argv[5] if len(sys.argv) > 5 else "dev"
# resdir = sys.argv[6] if len(sys.argv) > 6 else "res"
# max_batch = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
# use_ent = False  # Old parameter, unsuccessful test.
# model = sys.argv[8] if len(sys.argv) > 8 else "bert-large-cased"
# start_at = int(sys.argv[9]) if len(sys.argv) > 9 else 0
# stopfile = sys.argv[10] if len(sys.argv) > 10 else "stopper.txt"

# run_exp.sh 2 3 data docred dev res 1000 bert-large-cased 0 && docker logs --follow disbelief

# BIG NOTE: Assuming that this is located in your home directory. You may need to change the portion after -v and before : to the correct local folder.
docker run -d -i -v ~/iswc2025-217:/iswc2025-217 --gpus=all --name=disbelief $CONTAINER:latest bash -c "cd /iswc2025-217 &&
 python EntitySubstituteTest.py $1 $2 $3 $4 $5 $6 $7 $8 0"

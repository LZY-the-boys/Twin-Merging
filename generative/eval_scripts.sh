
SUITE="merged"
CONF=moe2
OUT=outs
name=CONF


function run_finetune(){
PORT=$(shuf -i7000-9000 -n1)

server_command="source activate merging;\
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
individual=$src \
uvicorn eval_merge:app --port $PORT | tee $name.log"

helm_command="SHOW=1 \
OUTPUT=$OUT \
NAME=$src \
SUITE=$SUITE \
CONF=$CONF \
PORT=$PORT \
bash helm.sh "

tmux_name=qwen-$src
tmux new-session -ds $tmux_name
tmux list-panes -t $tmux_name:0.1 > /dev/null 2>&1
if [ $? -ne 0 ]; then

    tmux split-window -h -t $tmux_name:0.0
fi
tmux send-keys -t $tmux_name:0.0 "$server_command" C-m
tmux send-keys -t $tmux_name:0.1 "$helm_command" C-m
# tmux a -t $tmux_name

# kill_process_by_name "eval_merge:app --port $PORT"
}


function eval_merged(){

PORT=$(shuf -i7000-9000 -n1)

server_command="source activate merging;\
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
select_merge=$select_merge \
src_merge=$src_merge \
base_model=$base_model \
yaml_file=/home/LeiFeng/lzy/lora-merge/dare/config/$merge.yml \
uvicorn eval_merge:app --port $PORT | tee $name.log"

helm_command="SHOW=1 \
OUTPUT=$OUT \
NAME=$select_merge-$merge \
SUITE=$SUITE \
CONF=$CONF \
PORT=$PORT \
bash helm.sh "

tmux_name=$select_merge-$merge-$info
tmux new-session -ds $tmux_name
tmux list-panes -t $tmux_name:0.1 > /dev/null 2>&1
if [ $? -ne 0 ]; then

    tmux split-window -h -t $tmux_name:0.0
fi
tmux send-keys -t $tmux_name:0.0 "$server_command" C-m
tmux send-keys -t $tmux_name:0.1 "$helm_command" C-m
tmux a -t $tmux_name

}


function run_pretrained(){
PORT=$(shuf -i7000-9000 -n1)

server_command="source activate merging;\
pretrained=1 \
base_model=$base_model \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
uvicorn eval_merge:app --port $PORT | tee $name.log"

# : ${SUITE:=$(date '+%Y-%m-%d-%M')}
base_model="${base_model//\//}"

helm_command="SHOW=1 \
OUTPUT=$OUT \
NAME=$base_model \
SUITE=$SUITE \
CONF=$CONF \
PORT=$PORT \
bash helm.sh "

# tmux_name=pretrained
tmux_name=$base_model
tmux new-session -ds $tmux_name
tmux list-panes -t $tmux_name:0.1 > /dev/null 2>&1
if [ $? -ne 0 ]; then

    tmux split-window -h -t $tmux_name:0.0
fi
tmux send-keys -t $tmux_name:0.0 "$server_command" C-m
tmux send-keys -t $tmux_name:0.1 "$helm_command" C-m
tmux a -t $tmux_name

}

function gen_eval_data(){

PORT=$(shuf -i7000-9000 -n1)

server_command="source activate merging;\
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
uvicorn gen_eval_data:app --port $PORT | tee $name.log"

helm_command="SHOW=1 \
OUTPUT=outs \
NAME=twin-$new_rank \
SUITE=$SUITE \
CONF=$CONF \
PORT=$PORT \
bash helm.sh "

tmux_name=helm-$new_rank-$info
echo "$PORT-$tmux_name"
tmux new-session -ds $tmux_name
tmux list-panes -t $tmux_name:0.1 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    tmux split-window -h -t $tmux_name:0.0
fi
tmux send-keys -t $tmux_name:0.0 "$server_command" C-m
tmux send-keys -t $tmux_name:0.1 "$helm_command" C-m
tmux a -t $tmux_name

}


function run_twin(){

PORT=$(shuf -i7000-9000 -n1)

data_path="data/test_router.json"

server_command="source activate merging;\
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
new_rank=$new_rank \
ablation=$ablation \
data_path=$data_path \
src_twin=$src_twin \
base_model=$base_model \
yaml_file=$yaml_file \
src_merge=$src_merge \
select_twin=$select_twin \
select_merge=$select_merge \
uvicorn eval_twin:app --port $PORT | tee $name.log"

helm_command="SHOW=1 \
OUTPUT=$OUT \
NAME=twin-$new_rank \
SUITE=$SUITE \
CONF=$CONF \
PORT=$PORT \
bash helm.sh "

tmux_name=qwen-twin-$new_rank-$info
echo "$PORT-$tmux_name"
tmux new-session -ds $tmux_name
tmux list-panes -t $tmux_name:0.1 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    tmux split-window -h -t $tmux_name:0.0
fi
tmux send-keys -t $tmux_name:0.0 "$server_command" C-m
tmux send-keys -t $tmux_name:0.1 "$helm_command" C-m
tmux a -t $tmux_name
}
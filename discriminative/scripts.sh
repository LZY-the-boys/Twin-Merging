
set -e pipefail

date_today=$(date '+%Y-%m-%d')
outdir=${outdir:="outs/merge_results"}
mkdir -p ${outdir}


models_name=(
"cola"
"sst2"
"mrpc"
"stsb"
"qqp"
"mnli"
"qnli"
"rte"
)
models_to_merge=()
for d in "${models_name[@]}"; do
models_to_merge+=(../roberta/$d/roberta-base_lr1e-05)
done
select_merge=${select_merge:="8"}


function pos(){

if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1 
fi
src_merge=("${models_name[@]:0:$select_merge}") 

echo ">>> merged from $select_merge tasks"
echo ">>> merge ${src_merge[@]}"

data_path="data/test.json"
}


function run_dare_task_arith(){

pos

for i in 0.7 ; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/dare_merge.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--mask-rate $i \
--outdir $outdir

done

}

function run_dare_tie(){

pos

for i in 0.7 0.8 0.9; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/dare_merge2.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--mask-rate $i \
--outdir $outdir

done

}


function run_avg_merge(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/average_merge.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--outdir $outdir


}

function run_tie(){

pos


for i in 0.9; do
for j in 0.7; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--yaml-file config/ties_merge.yml \
--data-path $data_path \
--exclude-param ".*classifier.*" ".*bias.*"  \
--mask-rate $i \
--scaling $j \
--outdir $outdir

done
done

}


function run_task_arith(){

pos


for j in 0.29; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/task_arithmetic.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--scaling $j \
--outdir $outdir \
--save-path "outs/task_arithmetic"

done

}

function ft(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--base-model 'roberta-base' \
--data-path $data_path \
--exclude-param ".*classifier.*" ".*bias.*" \
--outdir "outs/finetuned" 

}

function pretrain(){

pos

python run_merge.py \
--models-to-merge 'NONE' \
--models-name 'NONE' \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--base-model 'roberta-base' \
--outdir $outdir 

}


function twin_merge(){

yml='config/twin_merge.yml'
# NOTICE: we only select prefix 
select_merge=${select_merge:="8"}
select_twin=${select_twin:="8"}

if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1 
elif [ $select_twin -eq 1 ]; then
    datapath="data_glue/new_dataset2.json"
    if [ -z $src_twin ];then
        echo "please set \$src_twin!"
        exit 1
    fi
else
    datapath=data/test_router.json
    src_twin=("${models_name[@]:0:$select_twin}") 
    src_merge=("${models_name[@]:0:$select_merge}") 
fi

mask_strategy=${mask_strategy:="svd"}
mask_rate=${mask_rate:="0.9"}
echo ">>> use data_path $datapath"
echo ">>> use outdir $outdir"
echo ">>> merged from $select_merge tasks"
echo ">>> use twin vector from $select_twin tasks"
echo ">>> mask_rate $mask_rate; mask_strategy $mask_strategy"
echo ">>> use yml $yml"

python twin_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--data-path $datapath \
--src-merge ${src_merge[@]} \
--src-twin ${src_twin[@]} \
--yaml-file $yml \
--share-expert outs/task_arithmetic \
--exclude-param ".*classifier.*" ".*bias.*" \
--mask-rate $mask_rate \
--mask-strategy $mask_strategy \
--outdir $outdir 

}
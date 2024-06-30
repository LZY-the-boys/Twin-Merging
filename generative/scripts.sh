set -e pipefail

outdir=${outdir:="outs/qwen_merged"}
mkdir -p ${outdir}

models_to_merge=(
../qwen/qwen-mmlu
../qwen/qwen-truthfulqa
../qwen/qwen-bbq
../qwen/qwen-cnn
)

function run_avg_merge(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--base-model "Qwen/Qwen-14B" \
--yaml-file config/average_merge.yml \
--outdir $outdir \
--lora 'qwen_lora.json'

}

function run_dare_task_arith(){

pos

for i in 0.7 ; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--base-model "Qwen/Qwen-14B" \
--yaml-file config/dare_merge.yml \
--mask-rate $i \
--outdir $outdir \
--lora 'qwen_lora.json'

done

}

function run_task_arith(){

for j in 0.3; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--base-model "Qwen/Qwen-14B" \
--src-merge ${models_to_merge[@]} \
--yaml-file config/task_arithmetic.yml \
--scaling $j \
--outdir $outdir \
--lora 'qwen_lora.json'

done

}

function run_tie(){

pos


for i in 0.7; do
for j in 0.3; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--base-model "Qwen/Qwen-14B" \
--yaml-file config/ties_merge.yml \
--mask-rate $i \
--scaling $j \
--outdir $outdir \
--lora 'qwen_lora.json'

done
done

}


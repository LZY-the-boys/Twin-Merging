
export PYTHONPATH=.

source eval_scripts.sh

if [ ! -f "outs/test.json" ]; then
# 1.1 get test data first
gen_eval_data
fi

if [ ! -d "outs/qwen_merged" ]; then
# 1.2 get shared expert
bash run_merge.sh
fi

if [ ! -f "data/test_router.json" ]; then
# 2. get router train data 
torchrun --master-port 23451 --nnodes=1 --nproc_per_node=8 \
router.py \
--no-train \
--shared-expert outs/qwen_merged/merged
# 3. train router
python3 router.py --train
fi

# 4. run twin-merging
CUDA_VISIBLE_DEVICES=0 run_twin
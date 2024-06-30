
# detail functions to run different algorithm
source scripts.sh


if [ ! -d "outs/task_arithmetic" ]; then
# 1. get the shared expert
run_task_arith
fi

if [ ! -f "data/test_router.json" ]; then
# 2. gen router dataset
torchrun --master-port 23451 --nnodes=1 --nproc_per_node=2 \
router.py \
--no-train \
--shared-expert outs/task_arithmetic
# 3. train router
python3 router.py --train
fi

if [ ! -d "outs/finetuned" ]; then
# 4. need to get finetuned expert performance first for evaluation
ft
fi

# 5. run evaluation
twin_merge
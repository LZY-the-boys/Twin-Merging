eval "$(conda shell.bash hook)"
conda activate crfm-helm
set -e pipefail

cd HELM-Extended-Local

: ${PORT:=8080}
: ${SUITE:=tmp}
: ${NAME:=moe}
: ${OUTPUT:="outs/metrics"}

function wait_port_available() {
    local port="$1"
    while true; do
        if nc -z localhost $port; then
            echo "$port start"
            break
        fi
        sleep 5
    done
    sleep 1
}

CONF=run_moe2.conf

echo ">>> use $CONF"

wait_port_available $PORT

# hack to 127.0.0.1:8080
T=$(date +%s)

python -m helm.benchmark.run \
    --conf-paths $CONF \
    --suite $SUITE \
    --max-eval-instances 499 \
    --num-threads 1 \
    --name $NAME \
    --url "http://127.0.0.1:$PORT"

# write output to summary in the end
if [ "$SHOW" ];then
    python -m helm.benchmark.presentation.summarize --suite $SUITE
    python nips_metrics.py --suite $SUITE --output-path $OUTPUT
fi

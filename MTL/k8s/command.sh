export COMMAND="sleep 1000000"
# export COMMAND="cd /nas/people/lukasz_bala/reproducibility/ifsl/SIB; python3 main.py --config config/minires_1_baseline.yaml"
envsubst <job.yml | kubectl create -f -

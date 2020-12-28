# export COMMAND="sleep 100000"
export COMMAND="cd /nas/people/lukasz_bala/reproducibility/ifsl/SIB; python3 main.py --config config/{CONFIG_NAME} 
envsubst <job.yml | kubectl create -f -

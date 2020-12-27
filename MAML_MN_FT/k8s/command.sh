export COMMAND="sleep 100000"
# export COMMAND="cd /nas/people/lukasz_bala/reproducibility/ifsl/MAML_MN_FT; 
envsubst <job.yml | kubectl create -f -

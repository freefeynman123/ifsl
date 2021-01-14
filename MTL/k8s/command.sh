# export COMMAND="sleep 1000000"
export COMMAND="cd /nas/people/lukasz_bala/reproducibility/ifsl/MTL; pip3 install -U wandb==0.9.5; python3 main.py --config=mini_1_resnet_baseline --gpu=0 --num_workers=16"
envsubst <job.yml | kubectl create -f -

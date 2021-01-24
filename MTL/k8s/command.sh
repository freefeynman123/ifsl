# export COMMAND="sleep 1000000"
export COMMAND="cd /nas/people/lukasz_bala/reproducibility/ifsl/MTL; python3 main.py --config=mini_1_resnet_baseline --gpu=0 --num_workers=16 --require_index=True --save_artifacts_dataset=True --save_hacc=True --phase=meta_eval"
envsubst <job.yml | kubectl create -f -

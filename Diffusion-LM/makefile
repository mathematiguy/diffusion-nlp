REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
DOCKER_REGISTRY := mathematiguy
IMAGE := ${REPO_NAME}.sif
RUN ?= singularity exec ${FLAGS} --nv ${IMAGE}
FLAGS ?= -B $$(pwd):/code --pwd /code
SINGULARITY_ARGS ?=

.PHONY: build shell docker docker-push docker-pull enter enter-root

train_classifier:
	${RUN} python train_run.py --experiment e2e-tgt-tree --app "--init_emb diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_e2e --e2e_train datasets/e2e_data --n_embd 16 --learned_emb yes " --pretrained_model bert-base-uncased --epoch 6 --bsz 10

train_e2e_data:
	${RUN} bash -c 'cd improved-diffusion && python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000 --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821 --e2e_train ../datasets/e2e_data " --notes xstart_e2e'

train_rocstory:
	${RUN} bash -c 'cd improved-diffusion && python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000 --seed 101 --noise_schedule sqrt --in_channel 128 --modality roc --submit no --padding_mode pad --app "--predict_xstart True --training_mode e2e --vocab_size 11043 --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64'

jupyter:
	${RUN} jupyter lab --ip 0.0.0.0 --port=8888 --NotebookApp.password=$(shell singularity exec ${IMAGE} python -c "from notebook.auth import passwd; print(passwd('jupyter', 'sha1'))")

squeue:
	watch squeue --user=${USER}

REMOTE ?= cn-f001
push:
	rsync -rvahzP ${IMAGE} ${REMOTE}.server.mila.quebec:${SCRATCH}

build: ${IMAGE}
${IMAGE}:
	sudo singularity build ${IMAGE} ${SINGULARITY_ARGS} Singularity

${REPO_NAME}_sandbox: ${IMAGE}
	singularity build --sandbox $@ ${IMAGE}

sandbox: ${REPO_NAME}_sandbox
	sudo singularity shell ${FLAGS} --writable ${REPO_NAME}_sandbox

shell:
	singularity shell ${FLAGS} ${IMAGE} ${SINGULARITY_ARGS}

root-shell:
	sudo singularity shell ${FLAGS} ${IMAGE} ${SINGULARITY_ARGS}

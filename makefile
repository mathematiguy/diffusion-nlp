REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
DOCKER_REGISTRY := mathematiguy
IMAGE := ${REPO_NAME}.sif
RUN ?= singularity exec ${FLAGS} --nv ${IMAGE}
FLAGS ?= 
SINGULARITY_ARGS ?=

.PHONY: sandbox container shell root-shell docker docker-push docker-pull enter enter-root

jupyter:
	${RUN} jupyter lab --ip 0.0.0.0 --port=8888 --NotebookApp.password=$(shell singularity exec ${IMAGE} python -c "from notebook.auth import passwd; print(passwd('jupyter', 'sha1'))")

SERVER ?= cn-f001
push:
	rsync -rvahzP ${IMAGE} ${SERVER}.server.mila.quebec:${SCRATCH}

${REPO_NAME}_sandbox: ${IMAGE}
	singularity build --sandbox ${REPO_NAME}_sandbox ${IMAGE}

sandbox: ${REPO_NAME}_sandbox
	sudo singularity shell ${FLAGS} --writable ${REPO_NAME}_sandbox

container: ${IMAGE}
${IMAGE}:
	sudo singularity build ${IMAGE} ${SINGULARITY_ARGS} Singularity

shell:
	singularity shell ${FLAGS} ${IMAGE} ${SINGULARITY_ARGS}

root-shell:
	sudo singularity shell ${FLAGS} ${IMAGE} ${SINGULARITY_ARGS}

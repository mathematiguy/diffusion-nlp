#! /bin/bash

set -x

source .env

export GIT_REPO=`git config --get remote.origin.url`
export GIT_BRANCH=`git branch --show-current .`

# Use git oauth token
git config --global url."https://${GIT_OAUTH_TOKEN}@github.com/".insteadOf git@github.com:
git config --global user.email "caleb.moses@mila.quebec"
git config --global user.name "caleb.moses"

# Set up the dvc cache
dvc config cache.type symlink
dvc cache dir /work/dvc-cache

if [ ${CLEAR_CACHE} ]; then dvc gc --workspace -f ; fi

# Pull the cache
dvc pull || true

# Checkout the dvc project
dvc checkout || true

# Run the project
dvc repro

# Update the remote cache
dvc push

# Update dvc.lock in git repo
git add dvc.lock
git commit -am "Update dvc.lock"
git remote add df ${GIT_REPO}
git push df ${GIT_BRANCH}

#!/usr/bin/env bash
# this script was taken from cs6476

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENVIRONMENT_FILE=environment.yml
echo "Using ${ENVIRONMENT_FILE} ..."

# Create library environment.
conda env create -f "${SCRIPT_DIR}/${ENVIRONMENT_FILE}" \
&& eval "$(conda shell.bash hook)" \
&& conda activate ratslam \
&& python -m pip install -e "$SCRIPT_DIR/.."
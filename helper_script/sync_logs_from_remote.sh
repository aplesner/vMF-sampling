#!/bin/bash

# Check if project_variables.sh is sourced
if [ "${PROJECT_NAME}" != "vMF-sampling" ]; then
    echo "project_variables.sh is not sourced"
    exit 1
fi

# Define the remote directory
REMOTE_DIR="${USER_NAME}@${REMOTE_SERVER}:${CODE_STORAGE_DIR}"

# Sync code from the remote server to the local machine
rsync -avz --progress \
    --exclude-from=.ignore_for_code_sync \
    "${REMOTE_DIR}/logs" ./logs

echo "Code synced from remote!"
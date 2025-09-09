#!/bin/bash

# Check if project_variables.sh is sourced
if [ "${PROJECT_NAME}" != "flash-attention-benchmark" ]; then
    echo "project_variables.sh is not sourced"
    exit 1
fi

# Define the remote directory
REMOTE_DIR="${USER_NAME}@${REMOTE_SERVER}:${CODE_STORAGE_DIR}"

# Use rsync to sync to remote
rsync -avz --progress \
    --exclude-from=.ignore_for_code_sync \
    --delete \
    ./ "$REMOTE_DIR"

echo "Code synced to remote!"

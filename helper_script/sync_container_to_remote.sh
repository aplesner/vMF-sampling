#! /bin/bash

# Check that the PROJECT_NAME is sourced
if [ "$PROJECT_NAME" != "vMF-sampling" ]; then
    echo "helper_scripts/project_variables.sh is not sourced"
    exit 1
fi

# Sync the singularity container to the remote directory
REMOTE_DIR="${USERNAME}@${REMOTE_SERVER}:${SINGULARITY_STORAGE_DIR}"

# Sync the container to the remote directory
rsync -av --info=progress3\
    ./singularity/ $REMOTE_DIR
echo "Container synced to remote directory"


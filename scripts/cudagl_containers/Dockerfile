# Upgrade packages in a cudagl container
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# copy the script into the docker context
# the script needs to already be in '.'
COPY cudagl_package_updater.sh .

RUN USER=root bash ./cudagl_package_updater.sh

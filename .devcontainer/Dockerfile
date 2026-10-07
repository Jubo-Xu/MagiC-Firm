# Inherit from the micro-blossom toolchain base image (pinned Rust, sbt/
# SpinalHDL, Verilator; built from the micro-blossom submodule's Dockerfile
# by the initializeCommand above). Same image mb_toolchain.py's headless
# container (mb-magicfirm, see setup_local.sh) uses — the devcontainer is
# the interactive/VS Code way into the identical environment.
FROM micro-blossom

# Extras for working in this repo's Python tooling inside the container
RUN apt-get update \
    && apt-get install -y python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

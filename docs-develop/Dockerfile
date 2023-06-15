ARG PYT_VER=22.05
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3

# Specify poetry version
ENV POETRY_VERSION=1.1.13

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Install pandoc
RUN conda install pandoc

# Cache dependencies
COPY pyproject.toml ./

# Install modulus and dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction

WORKDIR /docs


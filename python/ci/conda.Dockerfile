FROM continuumio/miniconda3

WORKDIR /omnisci

RUN apt-get update && apt-get install -y build-essential

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "omnisci-connector-dev", "/bin/bash", "-c"]

ENTRYPOINT pip install -e .; pytest -x tests

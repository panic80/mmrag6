# -----------------------------------------------------------------------------
# Dockerfile (lower‑case name to match docker‑compose.yml)                     |
# -----------------------------------------------------------------------------
# This image bundles the Flask‑based Mattermost slash‑command app together     |
# with the ingestion / query helpers (ingest_rag.py, query_rag.py).            |
# It installs all Python dependencies – including docling (fetched from GitHub)|
# – and their system prerequisites.                                            |
# -----------------------------------------------------------------------------

FROM python:3.11-slim AS base

# Avoid prompts during apt installs
ARG DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Install OS‑level packages needed to build Python wheels (lxml, etc.) and git
# for the docling dependency that is pulled straight from GitHub.
# -----------------------------------------------------------------------------

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libxml2-dev \
        libxslt1-dev \
        # optional but nice to have: ca-certificates & curl for HTTPS calls
        ca-certificates \
        curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Create app directory & install Python deps
# -----------------------------------------------------------------------------

WORKDIR /app

# Pre‑copy requirements to leverage Docker layer caching when only source
# files change.
COPY requirements.txt ./

RUN pip install --upgrade pip \
 && pip install -r requirements.txt 

# -----------------------------------------------------------------------------
# Copy the rest of the application code
# -----------------------------------------------------------------------------

COPY . .

# Default port used by the Flask server
EXPOSE 5000

# Entrypoint – can be overridden by docker‑compose.yml if needed
CMD ["python3", "server.py"]

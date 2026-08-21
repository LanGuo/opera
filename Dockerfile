FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Node 20 (Claude Code requires Node 18+; Ubuntu 22.04 apt ships Node 12)
RUN apt-get update && apt-get install -y curl git python3.11 python3.11-venv python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# App
WORKDIR /app
COPY pyproject.toml uv.lock .
RUN uv sync --extra dev

COPY . .

# Workspace volume — mounted at runtime so run artifacts persist outside container
VOLUME ["/workspace"]
ENV ALLOW_HUMAN_INPUT=false

# Default: verify scaffold with tests.
# For eval runs, override: docker run ... bash scripts/run_eval.sh [model]
CMD ["uv", "run", "pytest", "-v"]

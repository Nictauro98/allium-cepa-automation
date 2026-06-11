FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

# Dependency layer — cached unless pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group serving --no-install-project

# Source layer
COPY src/ ./src/
COPY app/ ./app/
RUN uv sync --frozen --no-dev --group serving

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

RUN pip3 install --no-cache-dir --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    libgl1-mesa-glx

RUN pip3 install streamlit

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN mkdir -p /home/appuser/.cache /home/appuser/.config \
    && chown -R appuser:appuser /home/appuser

WORKDIR /home/appuser/blueprint
COPY --chown=appuser:appuser . .

RUN chmod +x demo/run.sh

USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN mkdir -p /home/appuser/tmp && chmod 777 /home/appuser/tmp
ENV TMPDIR=/home/appuser/tmp

RUN pip3 install -e .

EXPOSE 8501
ENTRYPOINT ["./demo/run.sh"]
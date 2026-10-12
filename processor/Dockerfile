FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN JAVA_DIR=$(ls -d /usr/lib/jvm/java-21-openjdk-* | head -n 1) \
    && ln -sfn "$JAVA_DIR" /usr/lib/jvm/default-java

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["python", "-m", "src.pipeline"]

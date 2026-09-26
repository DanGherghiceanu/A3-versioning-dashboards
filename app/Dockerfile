FROM python:3.12-slim

# torch needs OpenMP at runtime (same fix as A2)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so editing app.py rebuilds in seconds.
COPY requirements-torch.txt requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY *.py ./

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]

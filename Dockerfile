FROM dolfinx/dolfinx:v0.9.0

WORKDIR /root/HomiCSx

# Install pip and system deps
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install HomiCSx
COPY . .
RUN pip3 install -e .

# Jupyter
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
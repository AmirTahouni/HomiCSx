FROM dolfinx/dolfinx:v0.9.0

WORKDIR /root/HomiCSx

# Install gmsh from apt (works on ARM)
RUN apt-get update && apt-get install -y \
    gmsh \
    python3-gmsh \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Install HomiCSx from local source
COPY . .
RUN pip3 install -e .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
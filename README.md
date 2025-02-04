Instead of the traditional PRML algorithm, we use the leading AI model for symbol detection.

## Docker environment setup

Build image with no code inside the image
```bash
docker build -f ./dockerfile -t pr_nn:cuda .
```

Run docker image with gpu
```bash
docker run --gpus all --name pr_nn -it -v xx:xx pr_nn:cuda bash
```
Or without gpu
```bash
docker run --name pr_nn -it -v xx:xx pr_nn:cuda bash
```
## Conda environment setup

Create conda env
```bash
conda create -n prnn python=3.12.8 pip=24.2
```

Activate conda env
```bash
conda activate prnn
```
Install requirements
```bash
pip install -r requirements.txt
```


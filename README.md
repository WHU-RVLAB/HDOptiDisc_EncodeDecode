Instead of the traditional Viterbi algorithm, we use the leading neural network model for symbol detection.

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

my_package/
    __init__.py
    module1.py
    module2.py
work_dir/
    test.py


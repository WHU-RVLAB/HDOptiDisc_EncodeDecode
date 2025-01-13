Instead of the traditional Viterbi algorithm, we use the leading neural network model for symbol detection.

## Docker environment setup

Build the docker image with the following command
```bash
## build image with no code inside the image
docker build -f ./dockerfile --target base -t pr_nn:cuda .
```

Run the docker image with the following command
```bash
## run the following command to mount the repo-folder
docker run --gpus all --name pr_nn -it -v xx:xx pr_nn:cuda bash
## or run the following command without gpu
docker run --name pr_nn -it -v xx:xx pr_nn:cuda bash
```
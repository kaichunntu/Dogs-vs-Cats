## Cat vs Dog

This repository aims to provide a training framework for classification tasks.
The example is a task of training a classifier to identify a cat or a dog.
The usage is list as below. Follow the steps
1. Setup enviroment
2. Download the dataset
3. Training a classifier

### Setup environment
There are two ways to build environment. The first is using the docker. Please check that you already have
docker installed.
```
cd misc/DockerImage
docker build -t training:pytorch .

docker run -d --name training -v [local path]:/workspace training:pytorch 
```

The second is using the pip to install related packages, suggest that you have python3.8.12
```
pip install -r requirements.txt
```

### Download cat-vs-dog dataset




### Training a classifier




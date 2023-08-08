## Cat vs Dog

This repository aims to provide a training framework for classification tasks.
Some optimized techiques are implemented in this repository, such as Test Time Augmentation.

Here is an example of a task which is training a classifier to identify a cat or a dog. 
The experiment results are

Training metrics

<img src=./result_images/loss.png width=250 height=200> <img src=./result_images/acc.png width=250 height=200>


Experiment results
|mdoel|TTA|train acc|val acc|val precision|val_recall|AUC|
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
|ResNeXt|False|0.984|0.9690|0.9638|0.9747|0.9959|
|ResNeXt|True |0.984|0.9700|0.9644|0.9760|0.9962|


The usage is list as below. Follow the steps
1. Setup enviroment
2. Download the dataset
3. Training a classifier
4. Evaluate the classifier

### Setup environment
There are two ways to build environment. The first is using the docker. Please check that you already have
docker installed.
```
git clone 
cd misc/DockerImage
docker build -t training:pytorch .

docker run -d --name training -v [local path]:/workspace training:pytorch 
```

The second is using the pip to install related packages, suggest that you have python3.8.12
```
pip install -r requirements.txt
```
Note: Strongly recommended using docker.

### Download cat-vs-dog dataset
Download the dataset from [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats). unzip it to data folder.

### Training a classifier
First, complete the config. please follow the rule in config/readme.md.
Second, execute the command to train a classifier and save weights and config to the log directory.
```
python train.py config/model_example.yaml config/hyp.yaml --log_dir [log path]
```

### Evaluate the classifier
you can evaluate the classifier by running the following:
for reproducing the training accuracy
```
python eval.py runs/exp0/model_example.yaml runs/exp0/hyp.yaml --weights runs/exp0/weights/best_model.pt --save_dir results
```
for evaluating other dataset with ground truth
```
python eval.py runs/exp0/model_example.yaml runs/exp0/hyp.yaml --weights runs/exp0/weights/best_model.pt --data_root other/dataset/path --save_dir results
```
Or, if you want to play kaggle game, you can use the following to save predictions to a csv file
```
python inference.py ./data/test1 runs/exp0/model_example.yaml ./runs/exp0/hyp.yaml --weights runs/exp0/weights/best_model.pt --batch_size 32 --save_dir results
```

In addition, test time augment is an model free ensemble method. you can use '--use_tta' flag to activate this policy
```
python eval.py runs/exp0/model_example.yaml runs/exp0/hyp.yaml --weights runs/exp0/weights/best_model.pt --data_root other/dataset/path --save_dir results --use_tta
```





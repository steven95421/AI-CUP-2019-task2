# AICUP - TASK2
## team: SDML_SimpleBaseline (1st place)

### Requirements 環境
***
+ OS: Ubuntu 18.04
+ Python: 3.6.8
+ torch: 1.2.0
+ transformers: 2.0.0
+ sklearn: 0.22
+ [scibert](https://github.com/allenai/sciber)
### 資料處理
使用StratifiedShuffleSplit切train test 9:1，只使用abstract作為模型的輸入，
### Model
***
![](https://i.imgur.com/4ywFy27.png)

+ Sentence Encoder
    + SciBERT (scivocab, uncased)
    + Only get `[CLS]` embedding as representation of a whole sentence
+ Classifier
    + Linear Layer (768 -> 384)
        + Dropout (p=0.2)
        + Activation Function: gelu
        $$
        GELU(x)=x\cdot\Phi(x)
        $$
    + Linear Layer (384 -> 4)
        + Dropout (p=0.2)
        + Activation Function: Sigmoid
+ Training Procedure
    + Optimizer: Adam with linear warmup
    + Learning Rate: 1e-5
    + Batch Size: 32
    + Epoch: 3
    + Train : Test = 0.9 : 0.1
    + Loss Function: Binary Cross Entropy
    + Positive weight = [1.0, 1.0, 1.75, 7.5]
+ With above structure and parameters, we pass baseline with F1 score: 0.70793

### Keys for improvement
***
+ Quality of training data
    + Just change the random seed
    + Vary dramatically with different training data
+ Thresholds for classifying 1 and 0
    + Easily overfit local validation set
    + As for how to tune them we leave it to `post-process` part
+ Since local validation score is not reliable, we upload to TBrain to test and keep the predictions with good performance
    + Finally we collect 41 predictions

### Failed Attempts
***
+ Worsen Dramatically
    + Pre-trained Model (without tuning thresholds)
        + RoBERTa: **0.65**
            + Possible reason: trained on more CommonCrawl News （76 GB），Web （38GB）not specific for scientific paper 
        + XLNet: **0.68**
    + Classifier (without tuning thresholds)
        + LSTM: **0.678**
        + LSTM with attention: **<< 0.678**
        + XGBoost: **<< 0.678**
    + Training Method
        + Combine 4 different model: **0.607**
            + Possible reason: overfitting too much
            + Deprecated since it's too time-consuming 
+ No Influence
    + Classifier
        + Different input size of second linear layer
        + Different probability of dropout
    + Training Data
        + Pre-training on testing data
            + Possible reason: overfit on testing data, need to include training data too
        + Title concatenation
        + Separated Pre-trained for Title 
        + Category embedding concatenation
            + Prior probability embedding
        + Node embedding
            + Possible reason: citation per paper (in training data): 0.2
    + Training Method
        + Using only first 3 targets
            + If they are all 0, then fill `OTHERS` with 1; else 0

### Post-process
***
+ Straightforward approach
    + Tune thresholds for each category
        + Best public F1 score w/o ensembling: **0.729**
        + Thresholds: **[0.35, 0.3, 0.25, 0.35]**
        + F1 score w/ ensembling: **0.733**
+ Advanced work for post-processing
    + Tune single threshold for every category, which is slightly higher than thresholds of above approach. Then flip prediction of 0s if predicted logits are greatest among the 4 categories but not pass the threshold
        + In this manner we can improve the quality of 1s intuitively
        + Best public F1 score w/o ensembling: **0.733**
        + Threshold: **0.35**
        + F1 score w/ ensembling: **0.735**

### Reproducibility
***
+ Train
    + `python bert_finetune.py [seed] [gpu_id]`
        + seed: for both data spliting and torch
+ Predict
    + `python predict.py [seed] [gpu_id]`
        + seed: load model trained by this seed
+ Ensemble
    + run all cells directly in `vote_task2.ipynb`
        + which will vote with all predictions
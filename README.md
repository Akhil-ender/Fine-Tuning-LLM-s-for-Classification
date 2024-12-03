# Fine-Tuning-LLM-s-for-Classification
This repository contains various Fine Tuned Language Models for Classification Tasks.

## Introduction

The goal of this assignment is to conduct three experiments using three distinct language models. The objective is to understand and compare the performance of each model on the same dataset.

The three different models that are used for this experiment are DistillBERT, Albert-base-V2, Google Flan-T5. 

## Data Preprocessing:

We need to import and split the labels and text column. Then we need to make sure our Labels are in float data type as the AutoModelForSequenceClassification needs the label vectors to calculate the loss using BCEWithLogitsLoss. 

Cleaning the Texts dataset:

We are going to use CustomSpacyPreprocessor using a small vocabulary English model i.e.,  ‘en_core_web_sm’.

The hugging face trainer framework will be utilized for fine tuning the models. So we need to prepare our dataset into format them to match the input expectations of Hugging Face Trainer and Our models. We will split the given train dataset into validation trainset of 20% from the entire train dataset. We now have traindataset, validdataset, testdataset.


## Model Trainings

### Distilbert Training:
 We will need to create a compute metrics function to evaluate and get the f1 scores and accuracy.

We will use DistillBERTForSequenceClassification model from the 
For training the model we are going to use Adam optimizer with Momentum and a learning rate 3e-5, with a epoch of 10 and batch_size of 8. We will be saving the best model from the 
Training.
We will also use early stopping metric to stop the training if there is no signifant improvement in terms of model’s performance. With no significant increase of 0.1 for 5 continous epochs. This is done using EarlyStoppingCallback package from transformers.

DistillBERT performance

Since this is a multilabel classification problem we will be using Accuracy score as we want to be as accurate as possible when trying to classify the tweets into corresponding labels.

The best performing models was at the step 1100 with a accuracy of 0.844 with a validation loss of 0.37. We will log the results of the model to Weights&Biases dashboard and save the model. We will then use Pipeline package from Transformers for model inference and we will feed the pipeline with saved model, tokenizer and config for working on new task.

### Albert-Base-V2 Training:
We will need to create a compute metrics function to evaluate and get the f1 scores and accuracy.

We will use AlbertForSequenceClassification model from the 
For training the model we are going to use Adam optimizer with Momentum and a learning rate 3e-5, with a epoch of 10 and batch_size of 8. We will be saving the best model from the 
Training.
We will also use early stopping metric to stop the training if there is no signifant improvement in terms of model’s performance. With no significant increase of 0.1 for 5 continous epochs. This is done using EarlyStoppingCallback package from transformers.

Albert-Base-V2 performance:

Since this is a multilabel classification problem we will be using Accuracy score as we want to be as accurate as possible when trying to classify the tweets into corresponding labels.

The best performing models was at the step 250 with a accuracy of 0.85 with a validation loss of 0.36. We will log the results of the model to Weights&Biases dashboard and save the model. We will then use Pipeline package from Transformers for model inference and we will feed the pipeline with saved model, tokenizer and config for working on new task.

Albert-Base-V2 Training:

 We will need to create a compute metrics function to evaluate and get the f1 scores and accuracy.

We are going to use AlbertForSequenceClassification model. Training the model will be done using Adam optimizer with Momentum and a learning rate 3e-5, with a epoch of 10 and batch_size of 8. We will be saving the best model from the Training.
We will also use early stopping metric to stop the training if there is no signifant improvement in terms of model’s performance. With no significant increase of 0.1 for 5 continous epochs. This is done using EarlyStoppingCallback package from transformers.

### Flan-T5 performance:

Since this is a multilabel classification problem we will be using Accuracy score as we want to be as accurate as possible when trying to classify the tweets into corresponding labels.

The best performing models was at the step 300 with a accuracy of 0.13 with a validation loss of 0.44. We will log the results of the model to Weights&Biases dashboard and save the model. We will then use Pipeline package from Transformers for model inference and we will feed the pipeline with saved model, tokenizer and config for working on new task.

Difficulties faced during training this model is that it its shear size is computationally very expensive on the disk. This model was needed to be trained on a batches of size 4, even with a batch size of that size it required almost 12GB GPU RAM. The other difficulty was that the model took a lot of time and lot of re training to converge.

## Model Comparision:

Once we have trained all the three model we will now use this model to train unseen data to classify and to keep the testing and evaluation fair we are going to use same sentences for Classifying that texts into one of the 11 Labels.


From the test's we see that DistillBert uncased base and Albert-Base-V2 have given the same outputs with Albert-Base-V2 being slightly more accurate than previous one. Now for the Flan-T5 model we can see that the model is predicting different values than that of the other two models, the reason for this is that the Flan-T5 is much larger and complex model which requires a lot more data for it to be trained accurately for it to make right proper predictions.

## Summary

The selected three models are interpretatively different but have been finetuned for our task that is Tweet Classification. Each model has its own merits and demerits which can be classified based on various categories which are discussed below:

Model’s Size: Given the selected model, Flan-T5 is a much larger model when compared to other two models. It has more parameters and is computationally more expensive when compared to the other two models. This was one of the difficulty that was faced during model training is that Flan-T5 model required more GPU than the other two. This model was trained on a batch size of 4 which is the best possible batch size for the model to even train. Hence, this caused a much longer training time when compared to other two models.
Model’s Performance:
Among the three models, Albert-Base-V2 was the most accurate model and Flan-T5 being least accurate when. This finding was initially very shocking as I expected Flan-T5 to be most accurate given its size and parameters but this assumption was wrong and I think the model requires even more data for it to be trained so that it can predict right results.


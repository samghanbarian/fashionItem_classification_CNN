# fashionItem_classification_CNN


## Project Intro/Objective
The purpose of this project is to develop a CNN for classifying and labling fashion Items,

### Methods Used
* CNN
* Machine Learning
* Data Visualization
* Predictive Modeling
* etc.


## Files in Repo:
* .[primary classification.ipynb](fashionItem_classification_CNN/primary classification.ipynb) Contains implementation of CNN model to classify between three main classes: 'Glasses/Sunglasses', 'Trousers/Jeans', 'Shoes'
* .[Glasses and Sunglasses classification.ipynb](fashionItem_classification_CNN/Glasses and Sunglasses classification.ipynb) This file contains the code to classify the glasses into 'optical' and 'sun' glasses.
* .[]()


## Project Description
The project is to  identify and lable fashion item pictures collected from the web. The dataset composed of 16000 colored pictures 300 by 400 pixels in size. The pictures only featuring the items not the models. The items including shoes, glasses and sunglasses and trousers and jeans for both male and female.

There are three main categories 'Shoes','Glasses/sunglasses' and 'trousers/jeans'. For glasses/sunglasses category,the model should label each one as 'Optical' or 'Sunglasses'. Regarding trousers and jeans, beside splitting into 'trosers' and 'jeans' subcategories,The model should identify the gender as well. so in this category there are 4 subcategories: 'Feamle Jeans', 'Female trousers', 'Male Jeans' and 'Male trousers'. The shoes class contains 11 subcategories. for males there are 'Boots', 'Trainers/sneakers','Sandals','Formal shoes' and 'Other'. The subcategories regarding the female shoes are: 'Boots', 'Ballerina shoes','Sneakers/trainesr','Sandals','High heels' and 'Other'. For each subcategory the model should be able to label the item cosidering gender 'male/female'. So given an image the model should classify for instance as "shoes-Female-sandals". 

### Preprocessing:
Before training the model the picture are resized to 90px by 120px for sake of computational power. To work with TensorFlow, we need to convert all the images into tensors.

image.thumbnail((90,120))
i_array = np.asarray(image)


For the pupose of creating and training the network, for each task a network is created. For the primary classification a network is defined to classify to 3 primary classes. teh output of this network then will be sent to another network for instance in trouser/jeans subcategory to label the items as "female/male"-"jeans/trousers".

### Hyperparameter definition and logging in Tensorboard
Since we will train the model with different parameters we need to log some metrics inorder to compare and the see which model will perform better. So Tensorboard is used for visualising and loging the metrics.


to log the hyperparameters, first we define hyperparameters and metric.
HP_FILTER_SIZE = hp.HParam('filter_size', hp.Discrete([3,5,7]))\n",
    "HP_NUM_FILTER = hp.HParam('filters_number', hp.Discrete([32,64,96,128]))\n"
#### Logging Hyperparameters   
"with tf.summary.create_file_writer(r'Logs/Model 1/hparam_tuning/').as_default():\n",
    "    hp.hparams_config(\n",
    "        hparams= [HP_FILTER_SIZE,HP_NUM_FILTER],\n",
    "        metrics = [hp.Metric(METRIC_ACCURACY,display_name='accuracy')]\n"
    
    def run(log_dir, hparams, session_num):\n",
    "    \n",
    "    with tf.summary.create_file_writer(log_dir).as_default():\n",
    "        hp.hparams(hparams)  # record the values used in this trial\n",
    "        accuracy = train_test_model(hparams, session_num)\n",
    "        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)\n"
   
(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

### Metric
For comparing different models, two metrics are considered: Accuracy and Confusion matrix.

### Network architecture



## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)





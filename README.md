## Detecting AI Authorship: Analyzing Descriptive Features for AI Detection

### Datasets
Datasets are split into 10 seperate files to reduce the posibility of timeout errors when prompting OpenAI API

### DatasetCreation.py
Contains methodology for prompting OpenAI API aswell as the list of random prompts used 

### DataUnderstanding&Modeling.py
Contains methodology for calculating features, including perplexity, grammar, n-gram distribution, type-token ratio, average token length, and frequency of function words as well as modelling of the two approaches


### Worst Classsified AI & Human Abstracts 
An excel file detailing the calculated features for the feature-based approach is made availiable to show the top 3 most uncertain predictions made by the model. 
The best classification based on probability is also availible for comparison


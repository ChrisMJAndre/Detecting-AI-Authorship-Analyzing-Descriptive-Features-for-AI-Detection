# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:24:59 2023

@author: chris
"""

import os 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import seaborn as sns
import matplotlib.pyplot as plt


from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')


# Read dataset

dataset1 = pd.read_pickle("data_seperated_gbt1.pkl")
dataset2 = pd.read_pickle("data_seperated_gbt2.pkl")
dataset3 = pd.read_pickle("data_seperated_gbt3.pkl")
dataset4 = pd.read_pickle("data_seperated_gbt4.pkl")
dataset5 = pd.read_pickle("data_seperated_gbt5.pkl")
dataset6 = pd.read_pickle("data_seperated_gbt6.pkl")
dataset7 = pd.read_pickle("data_seperated_gbt7.pkl")
dataset8 = pd.read_pickle("data_seperated_gbt8.pkl")
dataset9 = pd.read_pickle("data_seperated_gbt9.pkl")
dataset10 = pd.read_pickle("data_seperated_gbt10.pkl")


DatasetBig = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10],ignore_index=True)


#Drop date for now - perhaps in the real one, loot out all the ones after 2022? 
dataset = DatasetBig.drop(['update_date',], axis=1)

# Create a binary label for AI text (1) and human text (0)
dataset["is_ai_generated"] = dataset["ai_generated"].apply(lambda x: 1 if x else 0)

## DROP alt under 20 ord? 
## DROP alt som indeholder "no abstract" "withdrawn" 
## DATA CLEANING THROUGHTS 

removed = dataset[dataset['abstract'].str.contains('$',regex=False) | ((dataset['abstract'].str.contains('=',regex=False))) | ((dataset['abstract'].str.contains('^',regex=False)))]

removed = dataset[dataset['abstract'].str.contains('=',regex=False) & ((dataset['abstract'].str.contains('$',regex=False))!= True)& ((dataset['abstract'].str.contains('^',regex=False))!= True)]
removed = dataset[dataset['abstract'].str.contains('^',regex=False) & ((dataset['abstract'].str.contains('=',regex=False))!= True)& ((dataset['abstract'].str.contains('$',regex=False))!= True)]


removed = dataset[dataset['abstract'].str.contains('^',regex=False) & ((dataset['abstract'].str.contains('=',regex=False))!= True)& ((dataset['abstract'].str.contains('$',regex=False))!= True)]

removed = dataset[dataset['abstract'].str.contains('$',regex=False)]
removed = dataset[dataset['abstract'].str.contains('$',regex=False)]

removed = dataset[dataset['abstract'].str.len() < 150]


removed = dataset[(len(dataset['abstract']) < 25)==True]

removed = data[data.name.isin(list2) == False]




## maybe remove citations for human generated (et al)
## maybe remove $ and ^ if there are more than 5 in the text 
## Maybe remove = if there are more than 3 in the text 
## Consideration: we are using one chatbot to guess what the next word will be, there  might be a correlation between the two chatbots


#------------------------------------------ Perplexity ----------------------------------------------------#


#https://huggingface.co/docs/transformers/perplexity 
def calculate_perplexity_gpt2(text, tokenizer, model):
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad(): #Makes it process faster and with less memory  (does not track gradients)
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss #Returns the loss function, (a measure of how well it predicted the text)
        perplexity = torch.exp(loss).item() 
    return perplexity


def add_perplexity_column_huggingface(df):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    perplexities = [calculate_perplexity_gpt2(text, tokenizer, model) for text in df["abstract"]]
    df["Perplexity"] = perplexities

    return df

def add_perplexity_column_huggingface_1(df):
    print("Loading models")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    print("Finished Loading models")
    perplexities = []
    for i, text in enumerate(df["abstract"]):
        perplexity = calculate_perplexity_gpt2(text, tokenizer, model)
        perplexities.append(perplexity)
        print(f"Finished processing text {i+1}/{len(df)}")

    df["Perplexity"] = perplexities

    return df

dataset_with_perplexity = add_perplexity_column_huggingface_1(dataset)

print(dataset_with_perplexity.head())



# Create a box plot to visualize the distribution of perplexity scores
plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="is_ai_generated", y="Perplexity", data=dataset_with_perplexity,showfliers = False) ##REMOVED OUTLIERS!! 
#ax = sns.boxplot(x="is_ai_generated", y="Perplexity", data=dataset_with_perplexity) 
ax.set_xticklabels(["Human-generated", "AI-generated"])
ax.set_xlabel("Text")
ax.set_ylabel("Perplexity")
ax.set_title("Perplexity Distribution for Human-generated vs. AI-generated Text")

plt.show()



#---------------------------------- Statistics -------------------------------------------#
## Remove . and , 
## Stopwords (maybe)

def ngram_distribution(texts, max_ngram_length):
    all_ngrams = []
    for text in texts:
        tokens = word_tokenize(text)
        for n in range(1, max_ngram_length + 1):
            text_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(text_ngrams)
    return Counter(all_ngrams)

def descriptive_statistics(df, max_ngram_length=8):
    human_texts = df[df["is_ai_generated"] == 0]["abstract"]
    ai_texts = df[df["is_ai_generated"] == 1]["abstract"]
    
    return {
        "Human N-gram Distribution": ngram_distribution(human_texts, max_ngram_length),
        "AI N-gram Distribution": ngram_distribution(ai_texts, max_ngram_length),
        "Human Average Text Length": np.mean([len(word_tokenize(text)) for text in human_texts]),
        "AI Average Text Length": np.mean([len(word_tokenize(text)) for text in ai_texts]),
        "Human Average Text character Length": np.mean([len([*text]) for text in human_texts]),
        "AI Average Text character Length": np.mean([len([*text]) for text in ai_texts]),
        
        "Human Average Text character Length pr word": np.mean([len([*text]) for text in human_texts])/np.mean([len(word_tokenize(text)) for text in human_texts]),
        "AI Average Text character Length pr word": np.mean([len([*text]) for text in ai_texts])/np.mean([len(word_tokenize(text)) for text in ai_texts]),
        
    }



def plot_ngram_distribution(statistics, n):
    data = []
    for category in ["Human", "AI"]:
        ngrams = statistics[f"{category} N-gram Distribution"]
        counts = [count for ngram, count in ngrams.items() if len(ngram) == n]
        data.append(counts)

    fig, ax = plt.subplots()
    ax.boxplot(data, labels=[f"{category} {n}-grams" for category in ["Human", "AI"]])
    ax.set_title(f"{n}-gram Frequency Distribution")
    plt.show()

def plot_average_text_lengths(statistics):
    data = [statistics[f"{category} Average Text Length"] for category in ["Human", "AI"]]

    fig, ax = plt.subplots()
    ax.bar(["Human", "AI"], data)
    ax.set_ylabel("Average Text Length")
    ax.set_title("Comparison of Average Text Lengths")
    plt.show()
    
def plot_average_text_Character_lengths(statistics):
    data = [statistics[f"{category} Average Text character Length"] for category in ["Human", "AI"]]

    fig, ax = plt.subplots()
    ax.bar(["Human", "AI"], data)
    ax.set_ylabel("Average Text character Length")
    ax.set_title("Comparison of Average Text character Lengths")
    plt.show()

def create_top_grams_dataframes(stat, n):
    data = []
    for category in ["Human", "AI"]:
        ngrams = stat[f"{category} N-gram Distribution"]
        top_ngrams = [item for item in ngrams.items() if len(item[0]) == n]
        top_ngrams.sort(key=lambda x: x[1], reverse=True)
        top_ngrams = top_ngrams[:10]
        data.append(pd.DataFrame(top_ngrams, columns=['ngram', 'count']))
        data[-1]['type'] = category

    return data[0], data[1]

statistics = descriptive_statistics(dataset, max_ngram_length=8)

for n in range(1, 8):
    plot_ngram_distribution(statistics, n)

plot_average_text_lengths(statistics)
plot_average_text_Character_lengths(statistics)


for n in range(1, 8):
    human_df, ai_df = create_top_grams_dataframes(statistics, n)
    print(f"Top 10 {n}-grams for Human Text:")
    print(human_df)
    print(f"\nTop 10 {n}-grams for AI-generated Text:")
    print(ai_df)
    print("\n" + "="*80 + "\n")


def add_ngram_columns(dataset, max_ngram_length=3):
    for n in range(1, max_ngram_length + 1):
        dataset[f'{n}-gram Distribution'] = dataset['abstract'].apply(lambda text: ngram_distribution(text, n))
    return dataset

# -----------------------------------------------------Grammer -------------------------------------------------------#

# https://pypi.org/project/language-tool-python/
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
#https://spacy.io/usage/models#download-pip
#import spacy #Tried to get it into a spacy pipline for efficiency but could not get it to work. 
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def grammar_score(text):
    matches = tool.check(text)
    errors = len(matches)
    return errors / len(text.split())

def calculate_grammar_scores(df, text_column="abstract"):
    texts = df[text_column].tolist()
    grammar_scores = [grammar_score(text) for text in texts]
    df["Grammar Score"] = grammar_scores
    return df

df_with_grammar_scores = calculate_grammar_scores(dataset)



plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="is_ai_generated", y="Grammar Score", data=df_with_grammar_scores,showfliers = False) ##REMOVED OUTLIERS!! 
#ax = sns.boxplot(x="is_ai_generated", y="Grammar Score", data=df_with_grammar_scores)  
ax.set_xticklabels(["Human-generated", "AI-generated"])
ax.set_xlabel("Text Type")
ax.set_ylabel("Grammar Score")
ax.set_title("Grammar Score for Human-generated vs. AI-generated Text")

plt.show()


#-------------------------------------------- TTR -------------------------------------#

##TTR is the ratio obtained by dividing the types (the total number of different words) 
##occurring in a text or utterance by its tokens (the total number of words). 
##A high TTR indicates a high degree of lexical variation while a low TTR indicates the opposite.
##
##


def calculate_ttr(dataset):
    ttrs = []
    for index, row in dataset.iterrows():
        text = row['abstract']
        
        tokens = word_tokenize(text.lower())
        types = set(tokens) #create a set of unique tokens
        
        if len(tokens) == 0:
            ttrs.append(None)
        else:
            #The TTR score is calculated by dividing the number of types by the total number of tokens.
            ttr = len(types) / len(tokens)
            ttrs.append(ttr)
    
    return ttrs

def calculate_all_tokens(dataset):
    all_tokens = []
    for index, row in dataset.iterrows():
        text = row['abstract']
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)
    return all_tokens

def calculate_all_tokens_ngrams(dataset, n):
    all_ngrams = []
    for index, row in dataset.iterrows():
        text = row['abstract']
        tokens = word_tokenize(text.lower())
        text_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(text_ngrams)

    return all_ngrams



def calculate_ttr_all_texts(dataset):
    all_tokens = calculate_all_tokens(dataset)

    types = set(all_tokens)  # Create a set of unique tokens from all texts

    ttrs = []
    for index, row in dataset.iterrows():
        text = row['abstract']
        tokens = word_tokenize(text.lower())

        if len(tokens) == 0:
            ttrs.append(None)
        else:
            unique_token_count = sum(1 for token in types if token in tokens)
            ttr = unique_token_count / len(tokens)
            ttrs.append(ttr)
    
    return ttrs

def calculate_ttr_all_ngrams(dataset, n=2):
    all_ngrams = calculate_all_tokens_ngrams(dataset, n)

    types = set(all_ngrams)  # Create a set of unique n-grams from all texts

    ttrs = []
    for index, row in dataset.iterrows():
        text = row['abstract']
        tokens = word_tokenize(text.lower())
        text_ngrams = list(ngrams(tokens, n))

        if len(text_ngrams) == 0:
            ttrs.append(None)
        else:
            unique_ngram_count = sum(1 for ngram in types if ngram in text_ngrams)
            ttr = unique_ngram_count / len(text_ngrams)
            ttrs.append(ttr)
    
    return ttrs

def plot_ttr_histogram_ngrams(dataset, ai_generated, n=1):
    if ai_generated:
        color = 'red'
        title = f'Type-Token Ratio Histogram (AI-Generated Abstracts) - ngram={n}'
    else:
        color = 'green'
        title = f'Type-Token Ratio Histogram (Human-Generated Abstracts) - ngram={n}'

    ttrs = calculate_ttr_all_ngrams(dataset, n=n)

    plt.hist(ttrs, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('TTR')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_ttr_histogram_tokens(dataset, ai_generated):
    if ai_generated:
        color = 'red'
        title = f'Type-Token Ratio Histogram - tokens (AI-Generated Abstracts)'
    else:
        color = 'green'
        title = f'Type-Token Ratio Histogram -tokens (Human-Generated Abstracts)'

    ttrs = calculate_ttr_all_texts(dataset)

    plt.hist(ttrs, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel('TTR')
    plt.ylabel('Frequency')
    plt.show()    


# Plotting

human_abstracts = dataset.loc[dataset['is_ai_generated'] == False]
ai_abstracts = dataset.loc[dataset['is_ai_generated'] == True]

# Plot TTR histograms based on tokens
plot_ttr_histogram_tokens(human_abstracts, ai_generated=False)
plot_ttr_histogram_tokens(ai_abstracts, ai_generated=True)

# Plot TTR histograms based on n-grams (1-5)
for n in range(1, 6):
    print(f"Plots for n-grams (n={n})")
    plot_ttr_histogram_ngrams(human_abstracts, ai_generated=False, n=n)
    plot_ttr_histogram_ngrams(ai_abstracts, ai_generated=True, n=n)



# ------------------------------------------- SpaCy -------------------------------------------------------------#
#cython: auto_pickle=False 
#import sys; print(sys.implementation)

#import spacy

#nlp = spacy.load("en_core_web_sm")
#doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
#for token in doc:
#    print(token.text, token.pos_, token.dep_)


# ------------------------------------------- AVG Word Lenght -------------------------------------------------------------#

import pandas as pd
from nltk.tokenize import word_tokenize

def calculate_avg_word_length(dataset, ai_generated):
    filtered_dataset = dataset.loc[dataset['is_ai_generated'] == ai_generated]

    word_lengths = []
    for index, row in filtered_dataset.iterrows():
        text = row['abstract']
        #tokens = word_tokenize(text.lower()) #TOKENS
        tokens = text.split() # SPLIT by spaces 

        lengths = [len(token) for token in tokens]
        avg_length = sum(lengths) / len(lengths)
        word_lengths.append(avg_length)
    
    return word_lengths

human_avg_lengths = calculate_avg_word_length(dataset, ai_generated=False)
ai_avg_lengths = calculate_avg_word_length(dataset, ai_generated=True)

plt.hist(human_avg_lengths, bins=20, color='green', edgecolor='black')
plt.title('Average Word Length Histogram (Human-Generated Abstracts)')
plt.xlabel('Average Word Length')
plt.ylabel('Frequency')
plt.show()

plt.hist(ai_avg_lengths, bins=20, color='red', edgecolor='black')
plt.title('Average Word Length Histogram (AI-Generated Abstracts)')
plt.xlabel('Average Word Length')
plt.ylabel('Frequency')
plt.show()

#------------------------------------------------------ Modelling ---------------------------------------------------#

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
import string
import re
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Function for text normalization
def text_normalizer(texts, lemmatize=True, remove_stopwords=True, remove_digits=True, remove_common=True):
    # Initialize lemmatizer and stopwords
    #lemmatizer = WordNetLemmatizer()
    #stop_words = set(stopwords.words('english'))
    custom_stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'should', 'can', 'could', 'may', 'might', 'must', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they']
    normalized_texts = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove digits
        if remove_digits:
            text = re.sub(r'\d+', '', text)
        # Lemmatize
       # if lemmatize:
        #    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        # Remove stopwords
        if remove_stopwords:
            words = text.split()
            text = ' '.join([word for word in words if word not in custom_stopwords])
        if remove_common: 
            words = text.split()
            most_common = [word for word, count in Counter(words).most_common(10)]
            words = [word for word in words if word not in most_common]
            text = ' '.join(words)
        normalized_texts.append(text)
    return normalized_texts


# Split the data into training, validation, and testing sets
X = dataset["abstract"]
y = dataset["is_ai_generated"]

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)



# Create a pipeline
pipeline = Pipeline([
    ('text_normalizer', FunctionTransformer(text_normalizer, validate=False)),
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Define the parameter grid for GridSearchCV
param_grid = [
    {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(2, 2), (3, 3)],
        'classifier': [LogisticRegression()],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
    },
    {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(2, 2), (3, 3)],
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
    },
    {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(2, 2), (3, 3)],
        'classifier': [DummyClassifier()],
        'classifier__strategy': ['stratified', 'most_frequent', 'uniform']
    },
    {
        'tfidf__max_df': [0.9, 1.0],
        'tfidf__ngram_range': [(2, 2), (3, 3)],
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.1, 1, 10],
    }
]

# Perform a grid search with cross-validation #high presision
grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best combination of hyperparameters
print("Best parameters: ", grid_search.best_params_)

# Evaluate the model on Train data
y_train_pred = grid_search.predict(X_train)
print("Validation Classification Report:\n", classification_report(y_train, y_train_pred))

# Evaluate the model on  data
y_val_pred = grid_search.predict(X_val)
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

results = grid_search.cv_results_
# Initialize empty lists for each model
rf_params = []
lr_params = []
dummy_params = []
Multi_params = []

# Loop over each set of parameters in the GridSearchCV results
for i, params in enumerate(results["params"]):
    if isinstance(params["classifier"], RandomForestClassifier):
        # Add the best parameters for the Random Forest model to the list
        best_params = {"params": params, "mean_test_score": results["mean_test_score"][i]}
        rf_params.append(best_params)
    elif isinstance(params["classifier"], LogisticRegression):
        # Add the best parameters for the Logistic Regression model to the list
        best_params = {"params": params, "mean_test_score": results["mean_test_score"][i]}
        lr_params.append(best_params)
    elif isinstance(params["classifier"], DummyClassifier):
        # Add the best parameters for the Dummy Classifier model to the list
        best_params = {"params": params, "mean_test_score": results["mean_test_score"][i]}
        dummy_params.append(best_params)
    elif isinstance(params["classifier"], MultinomialNB):
        # Add the best parameters for the Dummy Classifier model to the list
        best_params = {"params": params, "mean_test_score": results["mean_test_score"][i]}
        Multi_params.append(best_params)

# Sort the best parameters by mean test score in descending order
rf_params = sorted(rf_params, key=lambda x: x["mean_test_score"], reverse=True)[:1]
lr_params = sorted(lr_params, key=lambda x: x["mean_test_score"], reverse=True)[:1]
dummy_params = sorted(dummy_params, key=lambda x: x["mean_test_score"], reverse=True)[:1]
Multi_params = sorted(Multi_params, key=lambda x: x["mean_test_score"], reverse=True)[:1]

# Create a DataFrame with the best parameters for each model
best_params_df = pd.DataFrame({
    "model": ["Random Forest", "Logistic Regression", "Dummy Classifier", "MultinomialNB"],
    "best_params": [rf_params[0]["params"], lr_params[0]["params"], dummy_params[0]["params"],Multi_params[0]["params"]],
    "mean_test_score": [rf_params[0]["mean_test_score"], lr_params[0]["mean_test_score"], dummy_params[0]["mean_test_score"], Multi_params[0]["mean_test_score"]]
})

print(best_params_df)

best_estimator = grid_search.best_estimator_
print(best_estimator.named_steps['classifier'])

# Extract the feature importances from the Random Forest Classifier
if isinstance(best_estimator.named_steps['classifier'], RandomForestClassifier):
    rf = best_estimator.named_steps['classifier']
    feature_importances = pd.Series(rf.feature_importances_, index=best_estimator.named_steps['tfidf'].get_feature_names_out())
    feature_importances.nlargest(20).plot(kind='barh')

if isinstance(best_estimator.named_steps['classifier'], LogisticRegression):
    lr = best_estimator.named_steps['classifier']
    feature_importances = pd.Series(lr.coef_[0], index=best_estimator.named_steps['tfidf'].get_feature_names_out())
    feature_importances.nlargest(20).plot(kind='barh')

if isinstance(best_estimator.named_steps['classifier'], MultinomialNB):
    nb = best_estimator.named_steps['classifier']
    feature_probabilities = pd.DataFrame(nb.feature_log_prob_.T, columns=['class_0', 'class_1'], index=best_estimator.named_steps['tfidf'].get_feature_names_out())
    feature_probabilities['diff'] = feature_probabilities['class_1'] - feature_probabilities['class_0']
    feature_probabilities.nlargest(20, 'diff').plot(kind='barh')

if isinstance(best_estimator.named_steps['classifier'], DummyClassifier):
    feature_importances = pd.Series([1], index=['dummy'])
    feature_importances.plot(kind='barh')
    
    







    
#--------------------------Distilbert-----------------------------------------------#



from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
# Train-test-validation split

train_df, test_val_df = train_test_split(dataset, test_size=0.3, stratify=dataset["is_ai_generated"], random_state=42)
val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df["is_ai_generated"], random_state=42)

# Convert the DataFrames to HuggingFace's Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(train_dataset.column_names)
print(train_dataset)

# Tokenize the input text
def tokenize(batch):
    return {
        **tokenizer(batch['abstract'], padding='max_length', truncation=True, max_length=512, return_tensors='pt'),
        'labels': batch['is_ai_generated']
    }



train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['abstract', 'ai_generated'])
val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=['abstract', 'ai_generated'])
test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['abstract', 'ai_generated'])

# Set the dataset format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model on validation data
val_predictions = trainer.predict(val_dataset)
val_preds = np.argmax(val_predictions.predictions, axis=1)
print("Validation Classification Report:\n", classification_report(y_val, val_preds))
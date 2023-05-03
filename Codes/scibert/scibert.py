#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[61]:


#path = "C:/Users/rusta/OneDrive/Desktop/ADA/Master Thesis/LitCovid/Dataset/"
#path = "/home/rustamtalibzade/Desktop/Thesis/csv_files/"


# In[62]:


# Load your dataset as a Pandas DataFrame
data = pd.read_csv("LitCovid_textlg.csv", index_col=0)


# In[63]:


data.drop(data[data['topic']=='Transmission'].index, inplace=True)
data = data.reset_index(drop=True)


# In[64]:


percentage = float(input("Enter (0-100)")) / 100.0


# In[65]:


# Calculate the proportions of each topic in the DataFrame
proportions = data['topic'].value_counts(normalize=True)

# Calculate the number of samples to keep for each topic based on the proportions
n_to_keep = (proportions * len(data) * percentage).round().astype(int)

# Group the DataFrame by the "topic" column
groups = data.groupby('topic')

# Sample the groups with the specified number of samples to keep
samples = []
for name, group in groups:
    if n_to_keep[name] > 0:
        samples.append(group.sample(n_to_keep[name], random_state=42))

# Concatenate the samples into a new DataFrame
data_sampled = pd.concat(samples)

# Shuffle the rows in the new DataFrame
data_sampled = data_sampled.sample(frac=1, random_state=42)


# In[66]:


data = data_sampled.reset_index(drop=True)


# In[67]:


labels = list(set(data['topic']))

#Create dictionary to map labels to integers
label_to_int = {label: i for i, label in enumerate(labels)}

# Create dictionary to map integers back to labels
int_to_label = {i: label for i, label in enumerate(labels)}

# Add integer representation column to dataframe
data['topic_int'] = data['topic'].apply(lambda x: label_to_int[x])

# Print label-to-integer correspondence
for label, integer in label_to_int.items():
    print(f"{label} -> {integer}")

# Print integer-to-label correspondence
for integer, label in int_to_label.items():
    print(f"{integer} -> {label}")


# In[68]:


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        abstract = self.data.iloc[idx]['text_lg']
        if not isinstance(abstract, str):
        	abstract = str(abstract)
        topic = self.data.iloc[idx]['topic_int']
        encoding = self.tokenizer(abstract, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(topic, dtype=torch.long)
        }

# Split the dataset into training and validation sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)  # 80% for training, 20% for testing
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)  # 64% for training, 16% for validation from the 80% train data


train_dataset = CustomDataset(train_df, tokenizer)
val_dataset = CustomDataset(val_df, tokenizer)


# In[71]:


test_df.to_csv("test_scibert.csv")


# In[69]:


# Define the model and the evaluation metric
model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=len(data.topic.unique()))
loss_function = CrossEntropyLoss()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=200,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    num_train_epochs=3,  # Number of epochs
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Set up the cross-validation
cv = StratifiedKFold(n_splits=5)

for train_index, val_index in cv.split(data, data['topic_int']):
    train_data = data.iloc[train_index]
    val_data = data.iloc[val_index]

    train_dataset = CustomDataset(train_data, tokenizer)
    val_dataset = CustomDataset(val_data, tokenizer)

    # Train and evaluate the model
    trainer.train_dataset = train_dataset
    trainer.eval_dataset = val_dataset
    trainer.train()
    trainer.evaluate()


# In[ ]:


# Save the model
model.save_pretrained("scibert_model")

'''
# In[ ]:


test_dataset = CustomDataset(test_df, tokenizer)


# In[ ]:


# Load the best model after training
model = BertForSequenceClassification.from_pretrained("your_model_directory")

# Evaluate the model on the test dataset
trainer.model = model
trainer.eval_dataset = test_dataset
eval_results = trainer.evaluate()


# In[ ]:



def get_predictions(model, tokenizer, dataset):
    predictions = []
    for item in dataset:
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = logits.argmax(dim=1).item()
        predictions.append(predicted_label)
    return predictions

y_true = test_df["topic"].tolist()
y_pred = get_predictions(model, tokenizer, test_dataset)

print(classification_report(y_true, y_pred))'''


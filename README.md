
# UCSC Engineering 2020

This project uses machine learning to predict the diagnoses of patients with congestive heart failure or patients with myocardial infarction. While it focuses primarily on these two conditions, the code can be abstracted to any number of diseases or conditions within [Synthea](https://github.com/synthetichealth/synthea).


## Data Generation

[Synthea](https://github.com/synthetichealth/synthea) is a tool for generating patient json files in FHIR format. The medical history of each patient is randomly generated using the modules supplied by Synthea. These modules can be viewed and edited with the [Synthea Module Builder](https://synthetichealth.github.io/module-builder/); however, we left these modules unchanged when generating our dataset.


## Data Pre-processing & Transformation

### Files

-  [synthea_data_pipeline.ipynb](https://github.com/anthem-ai/ucsc-engineering-2020/blob/master/Data%20Preprocessing%20and%20Tranformation/synthea_data_pipeline.ipynb)

### Brief Summary

```synthea_data_pipeline.ipynb``` is a jupyter notebook that takes a folder of Synthea generated json files as input. All relevant medical data is extracted for each patient and transformed into a machine learning model readable format called embeddings. These embeddings are exported as csv and npy files as output, ready to be sent to **Model Testing & Training**.

### Overview

[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcblxuU3ludGhlYV9kYXRhc2V0X2ZvbGRlciAtLT4gQ0hGRGF0YXNldFxuXG5TeW50aGVhX2RhdGFzZXRfZm9sZGVyIC0tPiBNWU9JTkZEYXRhc2V0XG5cbkNIRkRhdGFzZXQgLS0-IEVtYmVkZGluZ3NcblxuTVlPSU5GRGF0YXNldC0tPiBFbWJlZGRpbmdzXG5cbkVtYmVkZGluZ3MgLS0-fGdlbmVyYXRlX29uZWhvdHwgb3V0cHV0X2NzdlxuXG5FbWJlZGRpbmdzIC0tPnxnZW5lcmF0ZV93b3JkZW1ifCBvdXRwdXRfbnB5IiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcblxuU3ludGhlYV9kYXRhc2V0X2ZvbGRlciAtLT4gQ0hGRGF0YXNldFxuXG5TeW50aGVhX2RhdGFzZXRfZm9sZGVyIC0tPiBNWU9JTkZEYXRhc2V0XG5cbkNIRkRhdGFzZXQgLS0-IEVtYmVkZGluZ3NcblxuTVlPSU5GRGF0YXNldC0tPiBFbWJlZGRpbmdzXG5cbkVtYmVkZGluZ3MgLS0-fGdlbmVyYXRlX29uZWhvdHwgb3V0cHV0X2NzdlxuXG5FbWJlZGRpbmdzIC0tPnxnZW5lcmF0ZV93b3JkZW1ifCBvdXRwdXRfbnB5IiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)


## Model Testing & Training

### Files

-  [training.py](https://github.com/anthem-ai/ucsc-engineering-2020/blob/master/Model%20Training%20and%20Testing/training.py)
	- ```python3 training.py <folder with csvs> <label>```

-  [w2vtrain.py](https://github.com/anthem-ai/ucsc-engineering-2020/blob/master/Model%20Training%20and%20Testing/w2vtrain.py)
	- ```python3 w2vtrain.py <folder containing npy files> ```

### Brief Summary

The output csv and npy folders are used as input for ```training.py``` and ```w2vtrain.py```. Both scripts use various machine learning models to create a prediction accuracy.


### Overview

[![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcblxub3V0cHV0X2Nzdi0tPiB0cmFpbmluZy5weVxuXG50cmFpbmluZy5weS0tPiBGTk4oRk5OKVxuXG50cmFpbmluZy5weS0tPiBDTk4oQ05OKVxuXG4gIFxuXG5vdXRwdXRfbnB5LS0-IHcydnRyYWluLnB5XG5cbncydnRyYWluLnB5LS0-IEZOTncoRk5OKVxuXG53MnZ0cmFpbi5weS0tPiBDTk53KENOTilcblxudzJ2dHJhaW4ucHktLT4gUk5OdyhSTk4pXG4iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcblxub3V0cHV0X2Nzdi0tPiB0cmFpbmluZy5weVxuXG50cmFpbmluZy5weS0tPiBGTk4oRk5OKVxuXG50cmFpbmluZy5weS0tPiBDTk4oQ05OKVxuXG4gIFxuXG5vdXRwdXRfbnB5LS0-IHcydnRyYWluLnB5XG5cbncydnRyYWluLnB5LS0-IEZOTncoRk5OKVxuXG53MnZ0cmFpbi5weS0tPiBDTk53KENOTilcblxudzJ2dHJhaW4ucHktLT4gUk5OdyhSTk4pXG4iLCJtZXJtYWlkIjp7InRoZW1lIjoiZGVmYXVsdCJ9LCJ1cGRhdGVFZGl0b3IiOmZhbHNlfQ)

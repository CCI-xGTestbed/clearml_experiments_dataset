from clearml import Dataset, Task
from datasets import load_dataset
from clearml import Task

import matplotlib.pyplot as plt

#data loading code here
df=Dataset.create(dataset_name='radio_map_1', dataset_project='tf_project_1')
df.add_files(path="data/")

df.upload()
df.finalize()

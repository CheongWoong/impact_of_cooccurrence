# Impact of Co-occurrence on Factual Knowledge of Large Language Models (EMNLP 2023 Findings)
This is a repository for the paper "[Impact of Co-occurrence on Factual Knowledge of Large Language Models](https://aclanthology.org/2023.findings-emnlp.518.pdf)".  
The project page can be found [here](https://cheongwoong.github.io/projects/impact_of_cooccurrence/).

<p align="center">
<img src="https://github.com/CheongWoong/cheongwoong.github.io/blob/master/assets/img/publication_preview/impact_of_cooccurrence.png"></img>
</p>

## Installation

### Set up a Conda Environment
This setup script creates an environment named 'factual_knowledge_probing'.
```
bash scripts/installation/setup_conda.sh
```

### Download the Pile dataset
The dataset is saved in 'data/pile'.
```
bash scripts/installation/download_pile.sh
bash scripts/installation/extract_pile.sh
```

### (Optional) Download the LAMA TREx dataset
The original dataset is saved in 'data/original_LAMA'.  
The preprocessed dataset is saved in 'data/LAMA_TREx'.
```
bash scripts/installation/download_LAMA.sh
bash scripts/installation/preprocess_LAMA_TREx.sh
```

### Download the preprocessed datasets
The datasets LAMA\_TREx and ConceptNet are provided [here](https://github.com/CheongWoong/factual_knowledge_probing/tree/main/data).

Check the number of samples for each relation.
```
# dataset_name: ['LAMA_TREx', 'ConceptNet']
bash scripts/installation/check_number_of_samples.sh {dataset_name}
```


## Factual Knowledge Probing
Follow [this](https://github.com/CheongWoong/factual_knowledge_probing) to run the factual knowledge probing experiments.  


## Precompute Data Statistics

### Extract Entities in the LAMA TREx and Target Vocabularies
The outputs are saved in 'data_statistics/entity_set'.
```
bash scripts/data_statistics/precompute/extract_entity_set.sh {dataset_names}
```
For example, run the following command to extract entities from LAMA_TREx and ConceptNet.
```
bash scripts/data_statistics/precompute/extract_entity_set.sh "LAMA_TREx ConceptNet"
```

### Compute Term Document Index of Entities
The outputs are saved in 'data_statistics/term_document_index/{pretraining_dataset_name}'.  
In addition to pretraining_dataset_name, the name of the text file needs to be specified as the script processes each data chunk individually when the dataset is split into multiple chunks.
```
# pretraining_dataset_name: ['pile', 'bert_pretraining_data']
bash scripts/data_statistics/precompute/compute_term_document_index.sh {pretraining_dataset_name} {filename}
```

### Compute Cooccurrence Matrix
The outputs are saved in 'data_statistics/{pretraining_dataset_name}/cooccurrence_matrix'.
```
bash scripts/data_statistics/precompute/compute_cooccurrence_matrix.sh {pretraining_dataset_name} {filename}
```

### Aggregate Cooccurrence Matrix
The aggregated matrix is saved in 'data_statistics/cooccurrence_matrix/{pretraining_dataset_name}/cooccurrence_matrix.npy'.
```
bash scripts/data_statistics/precompute/aggregate_cooccurrence_matrix.sh {pretraining_dataset_name}
```


## Impact of Cooccurrence

### Term Frequency Baselines
The prediction files are saved in 'results/{baseline_name}'.
```
bash scripts/data_statistics/term_frequency_baselines/marginal_probability.sh
bash scripts/data_statistics/term_frequency_baselines/joint_probability.sh
bash scripts/data_statistics/term_frequency_baselines/PMI.sh
```

### Compute Hits@1 against Reciprocal Rank of Subject-Object Frequency
This evaluation script computes correlation and saves in 'hits_1_against_reciprocal_rank_{dataset_name}.json'.
```
bash scripts/data_statistics/correlation_analysis/hits_1_against_reciprocal_rank.sh {prediction_file}
```

### Compute Hits@1 against Conditional Probability of the Gold Object Given a Subject
This evaluation script computes correlation and saves in 'hits_1_against_conditional_prob_{dataset_name}.json'.
```
bash scripts/data_statistics/correlation_analysis/hits_1_against_conditional_prob.sh {prediction_file}
```

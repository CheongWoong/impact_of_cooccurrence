# Impact of Co-occurrence on Factual Knowledge of Large Language Models (EMNLP 2023 Findings)
This is a repository for the paper "[Impact of Co-occurrence on Factual Knowledge of Large Language Models](https://arxiv.org/abs/2310.08256)".  
The project page can be found [here](https://cheongwoong.github.io/projects/impact_of_cooccurrence/).

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

### Download the LAMA TREx dataset
The original dataset is saved in 'data/original_LAMA'.  
The preprocessed dataset is saved in 'data/LAMA_TREx'.
```
bash scripts/installation/download_LAMA.sh
bash scripts/installation/preprocess_LAMA_TREx.sh
```

Check the number of samples for each relation.
```
bash scripts/installation/check_number_of_samples_LAMA_TREx.sh
```


## Factual Knowledge Probing
Follow [this](https://github.com/CheongWoong/factual_knowledge_probing) to run the factual knowledge probing experiments.


## Precompute Data Statistics

### Extract Entities in the LAMA TREx and Target Vocabularies
The outputs are saved in 'data_statistics/entity_set'.
```
bash scripts/data_statistics/precompute/extract_LAMA_TREx_entity_set.sh
```

### Compute Term Document Index of Entities in the Pile
The outputs are saved in 'data_statistics/term_document_index'.
```
bash scripts/data_statistics/precompute/compute_term_document_index.sh
```

### Compute Cooccurrence Matrix
The outputs are saved in 'data_statistics/cooccurrence_matrix'.
```
bash scripts/data_statistics/precompute/compute_cooccurrence_matrix.sh
```

### Aggregate Cooccurrence Matrix
The aggregated matrix is saved in 'data_statistics/cooccurrence_matrix/cooccurrence_matrix.npy'.
```
bash scripts/data_statistics/precompute/aggregate_cooccurrence_matrix.sh
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

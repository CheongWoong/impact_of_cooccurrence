# Impact of Co-occurrence on Factual Knowledge of Large Language Models (EMNLP 2023 Findings)
This is a repository for the paper "[Why Do Neural Language Models Still Need Commonsense Knowledge?](https://arxiv.org/pdf/2209.00599.pdf)" and "[Impact of Co-occurrence on Factual Knowledge of Large Language Models](https://aclanthology.org/2023.findings-emnlp.518.pdf)" (EMNLP 2023 Findings) ([project page](https://cheongwoong.github.io/projects/impact_of_cooccurrence/)).

<p align="center">
<img src="https://github.com/CheongWoong/cheongwoong.github.io/blob/master/assets/img/publication_preview/impact_of_cooccurrence.png"></img>
</p>


## Installation

### Knowledge Probing
Follow [this](https://github.com/CheongWoong/factual_knowledge_probing) to run the knowledge probing experiments.  
This includes setting up a conda environment and knowledge probing datasets.

### Download the Pre-training Data (the Pile - No Longer Available)
The dataset is saved in 'data/pile'.
```
bash scripts/installation/download_pile.sh
bash scripts/installation/extract_pile.sh
```

For other datasets, place them in 'data/{dataset_name}'.


## Precompute Data Statistics

### Extract Entities in the Target Datasets and Model Vocabularies
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
The outputs are saved in 'data_statistics/cooccurrence_matrix/{pretraining_dataset_name}' and 'data_statistics/occurrence_matrix/{pretraining_dataset_name}'.
```
bash scripts/data_statistics/precompute/compute_cooccurrence_matrix.sh {pretraining_dataset_name} {filename}
bash scripts/data_statistics/precompute/aggregate_cooccurrence_matrix.sh {pretraining_dataset_name}

bash scripts/data_statistics/precompute/compute_occurrence_matrix.sh {pretraining_dataset_name} {filename}
bash scripts/data_statistics/precompute/aggregate_occurrence_matrix.sh {pretraining_dataset_name}
```

## Impact of Cooccurrence

### Term Frequency Baselines
The prediction files are saved in 'results/{baseline_name}/{pretraining_dataset_name}'.
```
bash scripts/data_statistics/term_frequency_baselines/marginal_probability.sh {pretraining_dataset_name} {dataset_name}
bash scripts/data_statistics/term_frequency_baselines/joint_probability.sh {pretraining_dataset_name} {dataset_name}
bash scripts/data_statistics/term_frequency_baselines/PMI.sh {pretraining_dataset_name} {dataset_name}
```

### Correlational Analysis between Co-occurrence and Knowledge Probing Accuracy
Refer to [ipython notebook](https://github.com/CheongWoong/impact_of_cooccurrence/tree/main/analysis/analysis_codes/cooccurrence) for correlation analysis.

### Analysis of MadeOf / Opposite Relations
Refer to [ipython notebook](https://github.com/CheongWoong/impact_of_cooccurrence/tree/main/analysis/analysis_codes/madeof) for analyzing the madeof relation.  
Refer to [ipython notebook](https://github.com/CheongWoong/impact_of_cooccurrence/tree/main/analysis/analysis_codes/opposite_relation) for analyzing two opposite relations.

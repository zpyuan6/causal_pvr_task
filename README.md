# Causal PVR Task
This is a task for evaluation of causal explanation methods and causal deep learning model through computer vision task (pointer value retrieval task).
## How to Start

### Env install

```
conda env create -n causal_pvr python==3.7.13
conda activate causal_pvr
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


### Code Run

1. Causal PVR task

- Original dataset generation
```
python data/causal_pvr_dataset_gheneration.py
```

- Training deep learning models as explained models
```
python model_training_for_causal_pvr_task.py
```

- Concept representation identification

```
python identify_concept_representation.py
```

- Causal structure identification and demonstration
```
python identify_causal_structure.py
```

- Post-hoc explanation generation
```
python 
```

- Evaluation of explanations and causality

2. MQNLI task

- Original dataset generation
```
python
```

- Training deep learning models as explained models
```
python 
```

- Concept dataset generation

- Concept representation identification

- Causal structure identification

- Post-hoc explanation generation

- Explanation evaluation

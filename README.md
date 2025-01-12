# MEXA-CTP-Mode-Experts-Cross-Attention-for-Clinical-Trial-Outcome-Prediction

Demo code for our proposed **MEXA-CTP** on Clinicla Trial Outcome Prediciton benchmark.

## Introduction
We proposed **MEXA-CTP** a new method that integrates multi-modal data based on the concept of mode experts, and is optimized with Cauchy and contrastive losses to capture the relevant drug- disease, drug-protocol, and disease-protocol interactions, without resorting to hand-crafted structures.

<div align=center><img src="https://github.com/murai-lab/MEXA-CTP-Mode-Experts-Cross-Attention-for-Clinical-Trial-Outcome-Prediction/blob/main/images/model.png" alt="MEXA-CTP model"/></div>

## Requirements
Create a new anaconda environment and install all required packages before runing the code.
```bash
conda create --name mexactp
conda activate mexactp
pip install requirements.txt
```
## Dataset
* Step1: Download dataset
  You can download the dataset [here](https://github.com/futianfan/clinical-trial-outcome-prediction).
* Step2: Prepare embeddings
```bash
# smiles embedding
python ./smiles_embedding/ensemble.py
python ./smiles_embedding/sembed.py
# icd embedding
python ./icd_embedding/ensemble.py
python ./icd_embedding/iembed.py
# criteria embedding
python ./criteria_embedding/cembed.py
```
* Step3: Tran/Validataion/Test spilt
```bash
python createDataset.py
```
Please note that changing the data path if necessary.

## Usage
Training
```bash
python main.py --job train --result_dir <path_to_folder>
```
You can find the best hypers under results-sota.

Testing
```bash
python main.py --job eval --model_path <path_to_folder>
```
To duplicate our results, please fix seed to 2023.


## Performance
Phase I evaluation results on Clinical Trial Outcome Prediction dataset.
| Method  | F1 | PR-AUC | ROC-AUC|
| ------------- | ------------- | ------------- | ------------- |
| LR | .495 | .513 | .485 |
| RF | .499 | .514 | .542 |
| KNN+RF | .621 | .513 | .528 |
| XGBoost | .624 | .594 | .539 |
| AdaBoost | .633 | .544 | .540 |
| FFNN | .634 | .576 | .550 |
| HINT | .598 | .581 | .573 |
| **MEXA-CTP** | **.713** | **.605** | **.593** |



Phase II evaluation results on Clinical Trial Outcome Prediction dataset.
| Method  | F1 | PR-AUC | ROC-AUC|
| ------------- | ------------- | ------------- | ------------- |
| LR | .527 | .560 | .559 |
| RF | .463 | .553 | .626 |
| KNN+RF | .624 | .573 | .560 |
| XGBoost | .552 | .585 | .630 |
| AdaBoost | .583 | .586 | .603 |
| FFNN | .564 | .589 | .610 |
| HINT | .635 | .607 | .621 |
| **MEXA-CTP** | **.695** | **.635** | **.538** |

Phase III evaluation results on Clinical Trial Outcome Prediction dataset.
| Method  | F1 | PR-AUC | ROC-AUC|
| ------------- | ------------- | ------------- | ------------- |
| LR | .624 | .553 | .600 |
| RF | .675 | .583 | .643 |
| KNN+RF | .670 | .587 | .643 |
| XGBoost | .694 | .627 | .668 |
| AdaBoost | .722 | .589 | .624 |
| FFNN | .625 | .572 | .620 |
| HINT | .814 | .603 | .685 |
| **MEXA-CTP** | **.857** | **.771** | **.693** |




## Advanced
You can customize the embedding method by inheriting the Family class under /utils/embeddings.py.  
You can train your own model with your custom hypers. To fine-tune your hyperparameters, follow our recommended strategy:
* Step 1: Dataset hypers
* Step 2: Training hypers
* Step 3: Model hypers
```bash
python main.py --job trian --hypers <hypers>
```
You can visualize the results by
```bash
python polt.py --path <path_to_folder>
```



## Publication
Please cite our papers if you use our idea or code:
```bash
@inproceedings{mexactp2025sdm,
  title={MEXA-CTP: Mode Experts Cross-Attention for Clinical Trial Outcome Prediction},
  author={Zhang, Yiqing and Xiaozhong, Liu and Murai, Fabricio},
  booktitle={Proceedings of the 2025 SIAM International Conference on Data Mining (SDM)},
  year={2025}
}
```

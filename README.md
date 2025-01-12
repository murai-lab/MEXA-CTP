# MEXA-CTP-Mode-Experts-Cross-Attention-for-Clinical-Trial-Outcome-Prediction

Demo code for our proposed **MEXA-CTP** on Clinicla Trial Outcome Prediciton benchmark.

## Introduction
We proposed **MEXA-CTP** a new method that integrates multi-modal data based on the concept of mode experts, and is optimized with Cauchy and contrastive losses to capture the relevant drug- disease, drug-protocol, and disease-protocol interactions, without resorting to hand-crafted structures.

<div align=center><img src="xxx/model.png" width="500" height="350" alt="MEXA-CTP model"/></div>

## Requirements
Create a new anaconda environment and install all required packages before runing the code.
```bash
conda create --name mexactp
conda activate mexactp
pip install requirements.txt
```
## Dataset
You can download the dataset [here]([https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography](https://github.com/futianfan/clinical-trial-outcome-prediction)). 

## Usage

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

Phase III evaluation results on Clinical Trial Outcome Prediction dataset.
| Method  | F1 | PR-AUC | ROC-AUC|
| ------------- | ------------- | ------------- | ------------- |




## Advanced

## Publication


## Publication
Please cite our papers if you use our idea or code:

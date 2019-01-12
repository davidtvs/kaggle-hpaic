# Human Protein Atlas Image Classification

This is the source code for my submission to the [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification) Kaggle competition. The final submission placed 265th (0.48371 - top 14%).

## 265th place solution (score: 0.48371)

An ensemble of 10 models:

- **Models:** 5 ResNet34 (5-fold), 2 ResNet18, 1 ResNet50, and 1 BNInception
- **Data:** kaggle and external data (HPA v18). Multilabel stratification using [iterative-stratification](https://github.com/trent-b/iterative-stratification) to create the data splits for the validation and folds
- **Image channels:** RGB
- **Image resolution:** 512x512 for all models
- **Training augmentations:** horizontal flip, vertical flip, transpose, random rotation (limited to 20°), color jitter
- **Test time augmentations:** horizontal flip, vertical flip, transpose, and brightness for a total of 15 augmentations. The predictions from the test dataset and TTA are ensemble by taking a weighted mean of the probabilities 
- **Optimizer:** Adam for the ResNet34 models and SGD with momentum for the remaining models
- **Loss function:** BCE (with logits)
- **Sampling strategy:** weighted samples using the inverse of the median class frequency (oversamples minority classes and undersamples majority classes)
- **Decision threshold search:** different decision thresholds are tried on the validation set and the best (highest score) single threshold and per class thresholds are chosen. Although, this ended up performing worse than using a decision threshold of 0.3 in the test set
- **Learning rate schedule:** reduce on metric plateau
- **Submission:** mean of predictions of all models. Samples for which there are no predictions are set to class `25`

### Replicating results

For each configuration file in the `config/ensemble/` folder do the following:
1. Train the models:
   ```sh
   python train.py -c config/ensemble/config_file.json
   ```
   Make sure `5f_r34_224.json` is trained before `5f_r34_512.json`, for the remaining the training order is irrelevant.
2. Run the script to find the best thresholds:
   ```sh
   python threshold_finder.py -c config/ensemble/config_file.json
   ```
3. Make predictions and create the submission files:
   ```sh
   python make_submission.py -c config/ensemble/config_file.json
   ```

The results will be saved in the directory `checkpoint/config_file/`. Inside, a folder will be created for each fold containing the checkpoint of the model and a history file with the loss and metrics. The model checkpoint contains the state of the trainer when the model reached the best validation score.

The decision thresholds found by the search script are also saved in the root of the directory. The submissions are stored in the `submissions` folder. As an example, after all the steps for `5f_r34_512.json`, the directory tree inside the `checkpoint/5f_r34_512/` directory will look like this:

```
.
├── fold_1
│   ├── model.pth
│   └── summary.json
├── fold_2
│   ├── model.pth
│   └── summary.json
├── fold_3
│   ├── model.pth
│   └── summary.json
├── fold_4
│   ├── model.pth
│   └── summary.json
├── fold_5
│   ├── model.pth
│   └── summary.json
├── submission
│   ├── ensemble_class_best.csv
│   ├── ensemble_class_best_fill25.csv
│   ├── ... (more csv files)
└── threshold.json
```

The final submission is an ensemble of the following files:
- `checkpoint/5f_r34_512/fold_1_lb.csv`
- `checkpoint/5f_r34_512/fold_2_lb.csv`
- `checkpoint/5f_r34_512/fold_3_lb.csv`
- `checkpoint/5f_r34_512/fold_4_lb.csv`
- `checkpoint/5f_r34_512/fold_5_lb.csv`
- `checkpoint/bn_512/fold_1_lb.csv`
- `checkpoint/r18_512/fold_1_lb.csv`
- `checkpoint/r18_512_logw/fold_1_lb.csv`
- `checkpoint/r50_512/fold_1_lb.csv`

Place the files in a directory and run:
```sh
python ensemble_csv.py -d path/to/directory/
```

## A better single-model solution (score: 0.49204)

A single ResNet50 that performs better than the ensemble above. This model wasn't selected as a final submission because it performed worse both in the validation set and the public leaderboard. Configuration:

- **Models:** ResNet50
- **Data:** kaggle and external data (HPA v18). Multilabel stratification using [iterative-stratification](https://github.com/trent-b/iterative-stratification) to create the data splits for the validation and folds
- **Image channels:** RGB
- **Image resolution:** 512x512 for all models
- **Training augmentations:** horizontal flip, vertical flip, transpose, random rotation (limited to 20°), color jitter
- **Test time augmentations:** horizontal flip, vertical flip, transpose, and brightness for a total of 15 augmentations. The predictions from the test dataset and TTA are ensemble by taking a weighted mean of the probabilities 
- **Optimizer:** Adam with learning rate 2e-5
- **Loss function:** BCE (with logits)
- **Sampling strategy:** weighted samples using the inverse of the median class frequency. The weights are also clipped between 1 and 5 which means that minority classes are only oversampled up to 5 times and majority classes are not undersampled as much
- **Decision threshold search:** different decision thresholds are tried on the validation set and the best (highest score) single threshold and per class thresholds are chosen. Although, this ended up performing worse than using a decision threshold of 0.3 in the test set
- **Learning rate schedule:** reduce on metric plateau
- **Submission:** samples for which there are no predictions are set to class `25`

### Replicating results

1. Train the model:
   ```sh
   python train.py -c config/best/single_r50_512.json
   ```
2. Run the script to find the best thresholds:
   ```sh
   python threshold_finder.py -c config/best/single_r50_512.json
   ```
3. Make predictions and create the submission files:
   ```sh
   python make_submission.py -c config/best/single_r50_512.json
   ```

The best submission is `fold_1_lb_fill25.csv`.

## Tried but didn't improve performance

- F1 loss
- Focal loss provided very similar results to standard BCE but took more time to converge
- Weighing the loss function to give more importance to the rare classes
- InceptionV3 struggled to converge and it was also rather heavy
- SEResNet50 struggled to converge and also increased training time by 30-50% when compared to a standard ResNet50

## Things that are likely to improve performance with minor changes

Pretraining 5-folds of ResNet50 (or SEResNet50) on lower resolution images, then fine-tuning on 512x512 images, and ensembling the 5 folds should yield a significant performance boost.

## Installation

### Dependencies:
- Python 3
- pip

### Installation:
1. Clone the repository
   ```
   git clone https://github.com/davidtvs/kaggle-hpaic.git
   ```
2. Install package requirements:
   ```
   pip install -r requirements.txt
   ```

### Dataset:
1. Download the dataset from Kaggle [here](https://www.kaggle.com/c/human-protein-atlas-image-classification/data)
2. Navigate to the `scripts` folder
3. Download the HPAv18 dataset as follows:
    ```sh
    python download_hpav18.py
    ```
    See the command-line arguments using the `-h` option.
4. The images will be downloaded to the `scripts` directory by default. This behaviour can be changed by moving the `scripts/HPAv18_train.csv` file to the desired location.
5. Zip the downloded images
6. Place the kaggle files and the HPAv18 files in the same directory. The directory tree should look like this:
    ```
    .
    ├── HPAv18_train.csv
    ├── HPAv18_train.zip
    ├── sample_submission.csv
    ├── test.zip
    ├── train.csv
    └── train.zip
    ```
7. Run the following command to convert the zip files to HDF5:
    ```sh
    python kaggle_hpav18_hdf5.py -s path/to/source/directory -d path/to/destination/directory
    ```
    See the command-line arguments using the `-h` option
8. (optional) The configuration files assume that the HDF5 files are in the following directory `../dataset/`.

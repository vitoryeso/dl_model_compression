# Deep Learning Model Compression

Implementation of some model compression techniques (using Pytorch):
- Pruning -> Quantization (over Epochs)
- Pruning -> Quantization (over Iterations)

Model training data format:
```bash
Model_1
|________metadata.json
|________training_1
|        |__________parameters.json
|        |__________best_weights.pth
|        |__________training_data.csv
|________training_2
|        |__________parameters.json
|        |__________best_weights.pth
|        |__________training_data.csv
|        ...
Model_2
|________metadata.json
|________training_1
|        |__________parameters.json
|        |__________best_weights.pth
|        |__________training_data.csv
|________training_2
|        |__________parameters.json
|        |__________best_weights.pth
|        |__________training_data.csv
|        ...
|...
```
- metadata.json (include dataset information, model architecture, library version, etc)
- parameters.json (learning rate, batch size, pruning strenght, bit width)
- training_data.csv (training loss, test loss, trainig accuracy, test accuracy, model density)

## Training
Install required libraries. If you want GPU acceleration, install CUDA first.

```bash
pip install torch torchvision pandas 
```
See train.py file. change some hyper parameters and check the paths

```bash
python train.py
```

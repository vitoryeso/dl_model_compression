from src.compression import pruning, quantization
from src.cifar10 import cifar10
from src.model import convModel
from src.utils import accuracy, density, init_bias_weights, plot_data
from src.utils import save_data_experiment, save_training_parameters, SaveBestModel
from hyper_parameters import get_param_comb

from statistics import mean as mean
import os
import torch
import torch.nn.functional as F
from torch import optim
import argparse
from tqdm import tqdm
import logging
import wandb
import hydra
from omegaconf import DictConfig

def fit_pruning(net, epochs, optimizer, gamma, b, learning_rate, loss_fn, train_dl, test_dl, early_stoppin=True):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    densities = []
    
    logging.info("Starting training...")
    
    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        train_acc = 0.0
        test_acc = 0.0

        compression = b > 0 and gamma > 0.0

        # setting the model to training mode
        net.train()

        # save pruning only model weights
        model_copy = type(net)()
        model_copy.to(dev)

        # TRAIN LOOP
        train_pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for i, data in enumerate(train_pbar):
            inputs, labels = data[0].to(dev), data[1].to(dev)

            # 1. First, calculate pruning masks (betas)
            betas = []
            if gamma > 0.0:
                betas = pruning(net, gamma)

            # 2. Save a copy of the original weights before compression
            model_copy.load_state_dict(net.state_dict())
            pesos_pruned = list(model_copy.parameters())

            # 3. Apply quantization (if enabled)
            if b > 0:
                quantization(net, b, betas)

            # 4. Forward pass with compressed weights
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            if compression:
                # Custom weight update using the original (uncompressed) weights
                with torch.no_grad():
                    for p_net, p_pruning in zip(net.parameters(), pesos_pruned):
                        torch.subtract(p_pruning, p_net.grad * learning_rate, out=p_net)
                    net.zero_grad()
            else:
                # Standard optimization if no compression
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()  
            train_acc += accuracy(outputs, labels).item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        if compression:
            betas = pruning(net, gamma)
            quantization(net, b, betas)

        net.eval()

        # TEST STEP
        test_pbar = tqdm(test_dl, desc=f'Epoch {epoch+1}/{epochs} [Test]')
        for i, data in enumerate(test_pbar):
            inputs, labels = data[0].to(dev), data[1].to(dev)

            with torch.no_grad():
                outputs = net(inputs)
            test_loss += loss_fn(outputs, labels).item()
            test_acc += accuracy(outputs, labels).item()

        train_loss /= len(train_dl)
        train_acc /= len(train_dl)
        test_loss /= len(test_dl)
        test_acc /= len(test_dl)
        dst = density(net)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        densities.append(density(net))
        
        logging.info(f'Epoch: {epoch + 1}/{epochs} | '
                    f'Train Loss: {train_loss:.3f} | '
                    f'Train Acc: {train_acc:.3f} | '
                    f'Test Acc: {test_acc:.3f} | '
                    f'Density: {dst:.3f}')

        save_best_model(test_acc, epoch, model)

        if epoch > 5:
            if early_stoppin and mean(train_accuracies[-5:-1]) <= 0.15:
                logging.info(f'Early Stopping at epoch: {epoch + 1}. Going to next parameter combination.')
                return train_losses, train_accuracies, test_losses, test_accuracies, densities

    logging.info('Finished Training')
    return train_losses, train_accuracies, test_losses, test_accuracies, densities

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize W&B
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        config=dict(cfg)
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {dev} device")

    if not os.path.exists(cfg.training.models_path):
        os.makedirs(cfg.training.models_path)

    dataset = cifar10(cfg.training.data_path, cfg.optimization.batch_size, data_augmentation=True)
    trainloader, testloader = dataset.get_loaders()
    del dataset

    loss_func = F.cross_entropy
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping_patience,
        min_delta=cfg.training.early_stopping_min_delta
    )

    model = convModel()
    model.to(dev)
    model.apply(init_bias_weights)
    opt = optim.SGD(model.parameters(), 
                    cfg.optimization.lr, 
                    momentum=cfg.optimization.momentum)

    training_metadata = fit_pruning(model, 
                                  cfg.training.epochs,
                                  opt,
                                  cfg.compression.gamma,
                                  cfg.compression.bits,
                                  cfg.optimization.lr,
                                  loss_func,
                                  trainloader,
                                  testloader)

    # ... rest of your training code ...

if __name__ == "__main__":
    main()

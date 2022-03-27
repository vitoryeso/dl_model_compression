from src.compression import pruning, quantization
from src.cifar10 import cifar10
from src.model import convModel

from src.utils import accuracy, density, init_bias_weights, plot_data
from src.utils import save_data_experiment, save_training_parameters, SaveBestModel
from hyper_parameters import get_param_comb

import os
import torch
import torch.nn.functional as F

from torch import optim

def fit_pruning(net, epochs, optimizer, gamma, b, learning_rate, loss_fn, train_dl, test_dl):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    densities = []

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
        for i, data in enumerate(train_dl):
            # get the inputs; data is a list of [inputs, labels]
            # passing input data to the dev(GPU) memory
            inputs, labels = data[0].to(dev), data[1].to(dev)

            betas = []
            # >>> PRUNING <<<
            if gamma>0.0:
                betas = pruning(net, gamma)

            # copy only pruninig weights
            model_copy.load_state_dict(net.state_dict())
            pesos_pruned = list(model_copy.parameters())

            # >>> QUANTIZATION <<<
            if b>0:
                quantization(net, b, betas)

            # forward
            outputs = net(inputs)
            
            # loss calculation
            loss = loss_fn(outputs, labels)
            
            # loss backward. grads are generated here
            loss.backward()
            
            # if no compression, perform default optimization
            if compression:
                with torch.no_grad():
                    for p_net, p_pruning in zip(net.parameters(), pesos_pruned):
                        #p_pruning.subtract(p_net.grad * 0.05)
                        #pesos = pesos_pruning - learning_rate*p_net.grad
                        torch.subtract(p_pruning, p_net.grad * learning_rate, out=p_net)
                    net.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()  
            train_acc += accuracy(outputs, labels).item()

        # final compression (after entire epoch)
        if compression:
            betas = pruning(net, gamma)
            quantization(net, b, betas)

        # change the model to the inference mode
        net.eval()

        # TEST STEP
        for i, data in enumerate(test_dl):
            inputs, labels = data[0].to(dev), data[1].to(dev)

            with torch.no_grad():
                outputs = net(inputs)
            test_loss += loss_fn(outputs, labels).item()
            test_acc += accuracy(outputs, labels).item()
            
        # getting model train metadata
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
        
        print('Epoch: %d/%d \
               train loss: %.3f \
               train accuracy: %.3f \
               test accuracy: %.3f \
               density: %.3f' %
            (epoch + 1, epochs, train_loss, train_acc, test_acc, dst))

        save_best_model(test_acc, epoch, model)
        

    print('Finished Training')
    return train_losses, train_accuracies, test_losses, test_accuracies, densities

if __name__ == "__main__":
    dev = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("USING ",dev," DEVICE!")
    batch_size = 256
    lr = 0.075
    models_path = "./models/model_cifar10/"

    data_path = "./data"

    dataset = cifar10(data_path, batch_size, data_augmentation=True)
    trainloader, testloader = dataset.get_loaders()
    del dataset;

    loss_func = F.cross_entropy

    hyper_parameters_comb = get_param_comb([0.0, 0.25, 0.5, 0.75],
                                       [32, 16, 8, 4, 3])

    count = 0
    EPOCHS = 200
    for i, parameter_set in enumerate(hyper_parameters_comb):
        for hyper_parameters in parameter_set:
            if count < 1:
                count += 1;
                continue;
            else: count += 1;
            
            lr, gamma, b = hyper_parameters
            print(f"""Starting a new train <<<<<<<<<<<<<<<<
                    GAMMA: {gamma},
                    BIT WIDTH: {b},
                    LEARNING RATE: {lr}
            """)
            metadata_path = models_path + f"training_{i + 1}/"

            save_best_model = SaveBestModel(metadata_path)

            if not os.path.isdir(metadata_path):
                os.mkdir(metadata_path)

            model = convModel()
            model.to(dev) # pass model to GPU memory
            model.apply(init_bias_weights) # init bias weights to avoid NaN
            #opt = optim.Adam(model.parameters(), lr=lr)
            opt = optim.SGD(model.parameters(), lr, momentum=0.9)

            #def fit_pruning(net, optimizer, epochs, gamma, b, pruning_rate, loss_fn, train_dl, test_dl):
            training_metadata = fit_pruning(model, 
                                            EPOCHS, 
                                            opt, 
                                            gamma, 
                                            b, 
                                            lr, 
                                            loss_func, 
                                            trainloader, testloader)

            #train_loss, test_loss, train_acc, test_acc, densities = training_metadata
            save_data_experiment(metadata_path, training_metadata)
            save_training_parameters(metadata_path,lr, batch_size, gamma, b)


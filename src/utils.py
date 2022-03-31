import torch
import json
import os
import pandas as pd

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def density(net):
    non_zero = 0
    count = 0
    with torch.no_grad():
        for p in net.parameters():
            non_zero += torch.count_nonzero(p).cpu().numpy()
            count += torch.numel(p)

    return non_zero / count

def init_bias_weights(m):
    try:
        m.bias.data.fill_(0.1)
    except: 
        pass

def plot_data(data, title="Training loss", xlabel="Iterations", ylabel="Loss"):
    x_axis = [i for i in range(len(losses))]
    plt.plot(x_axis,losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def save_data_experiment(save_path, log):
    train_losses, train_accuracies, test_losses, test_accuracies, densities = log

    df = pd.DataFrame({
        'train_loss': train_losses,
        'test_loss': test_losses,
        #'train_accuracy': list(map(lambda x: x.cpu().numpy(), train_accuracies)),
        #'test_accuracy': list(map(lambda x: x.cpu().numpy(), test_accuracies)),
        'train_accuracy': train_accuracies,
        'test_accuracy': test_accuracies,
        'density': densities
    })

    df['epoch'] = range(1, len(train_losses) + 1)
    df.to_csv(save_path + 'training_data.csv')

def save_training_parameters(save_path, learning_rate, batch_size, gamma, b):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    parameters = {
        "learning rate": learning_rate,
        "batch size": batch_size,
        "gamma": gamma,
        "bit witdh": b
    }

    with open(save_path + "parameters.json", "w+") as write_file:
        json.dump(parameters, write_file);


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation accuracy is bigger than the previous least acc, then save the
    model state.
    """
    def __init__(
        self, save_path: str, best_valid_acc=0.0
    ):
        self.save_path = save_path
        self.best_valid_acc = best_valid_acc
        
    def __call__(
        self, current_valid_acc, 
        epoch, model
    ):
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"\nBest validation accuracy: {self.best_valid_acc}")
            print(f"Saving best model for epoch: {epoch+1} in"
                    + self.save_path + "\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                }, os.path.join(self.save_path, "best_acc_model.pth"))


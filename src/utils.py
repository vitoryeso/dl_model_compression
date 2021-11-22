import torch

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

def save_data_experiment(save_path, net, log, parameters):
    train_losses, train_accuracies, test_losses, test_accuracies, densities = log

    df = pd.DataFrame({
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_accuracy': list(map(lambda x: x.cpu().numpy(), train_accuracies)),
        'test_accuracy': list(map(lambda x: x.cpu().numpy(), test_accuracies)),
        'density': densities
    })

    df['epoch'] = range(1, len(train_losses) + 1)
    df.to_csv(path + 'training_data.csv')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, save_path: str, best_valid_loss=float('inf')
    ):
        self.save_path = save_path
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'loss': criterion,
                }, self.save_path)

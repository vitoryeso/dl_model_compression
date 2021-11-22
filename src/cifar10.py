import torch
import torchvision
import torchvision.transforms as transforms

class cifar10:
    def __init__(self, root_path: str, batch_size: int, data_augmentation: bool):
        self.root_path = root_path
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        if data_augmentation:
            self.transform_train = transforms.Compose(
                  [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   # transforms Normalize ( (R_mean, G_mean, B_mean), (R_std, G_std, B_std) )
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
        else:
            self.transform_train = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_test = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_data(self):
        trainset = torchvision.datasets.CIFAR10(root=self.root_path, train=True,
                                                download=True, transform=self.transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root=self.root_path, train=False,
                                               download=True, transform=self.transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=0)
        return trainset, trainloader, testset, testloader

    def get_loaders(self):
        trainset = torchvision.datasets.CIFAR10(root=self.root_path, train=True,
                                                download=True, transform=self.transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root=self.root_path, train=False,
                                               download=True, transform=self.transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=0)
        return trainloader, testloader


    def get_testdata(self):
        testset = torchvision.datasets.CIFAR10(root=self.root_path, train=False,
                                               download=True, transform=self.transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=0)
        return testset, testloader

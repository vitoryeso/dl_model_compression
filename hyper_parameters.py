from itertools import product

# (learning_rate, gamma, b)
def get_param_comb(pruning_parameters, quantization_parameters):
    parameters = list(product(pruning_parameters,
                                quantization_parameters))
    print("HYPER PARAMETERS:")

    hyper_parameters = []
    for param in parameters:
        param_list = []
        #for lr in [0.01, 0.05, 0.07, 0.1]:
        for lr in [0.10, 0.05, 0.07, 0.09]:
            item = (lr, ) + param
            param_list.append(item)

        print(param_list)
        hyper_parameters.append(param_list)

    return hyper_parameters;

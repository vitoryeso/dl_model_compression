from itertools import product

# (learning_rate, gamma, b)
def get_param_comb(pruning_parameters, quantization_parameters):
    pruning_parameters = [0.25, 0.5, 0.75]
    quantization_parameters = [3, 4, 8, 16]

    parameters = list(product(pruning_parameters,
                                quantization_parameters))

    hyper_parameters = []
    for param in parameters:
        param_list = []
        for lr in [0.01, 0.05, 0.07, 0.1]:
            item = (lr, ) + param
            param_list.append(item)

        print(param_list)
        hyper_parameters.append(param_list)

    print(len(hyper_parameters))
    print(len(hyper_parameters[0]))


"""
parameters_1 = [
    (0.0001, 0.0, 0),
    (0.001, 0.0, 0),
    (0.01, 0.0, 0),
    (0.05, 0.0, 0)
]
parameters_2 = [
    (0.0001, 0.5, 8),
    (0.001, 0.5, 8),
    (0.01, 0.5, 8),
    (0.05, 0.5, 8),
    (0.07, 0.5, 8),
    (0.1, 0.5, 8)
]
parameters_3 = [
    (0.0001, 0.5, 4),
    (0.001, 0.5, 4),
    (0.01, 0.5, 4),
    (0.05, 0.5, 4),
    (0.07, 0.5, 4),
    (0.1, 0.5, 4)
]
parameters_4 = [
    (0.0001, 0.5, 3),
    (0.001, 0.5, 3),
    (0.01, 0.5, 3),
    (0.05, 0.5, 3),
    (0.07, 0.5, 3),
    (0.1, 0.5, 3)
]
parameter_configurations = [parameters_1, parameters_2, parameters_3, parameters_4 ]
"""

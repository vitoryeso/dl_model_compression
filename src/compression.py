import torch

def pruning(model, gamma):
    if gamma > 0.0:
        with torch.no_grad():
            pruning_thresholds = []
            # Aqui a gente pode calcular o desvio padrao da camada e também o desvio padrão da rede toda, ou por camadas mais estrategicamente escolhidas baseadas na arquitetura.
            # so fazer por camada e global, ja seria muito interessante.
            for p in model.parameters():
                # transformando o layer em um vetor unidimensional 
                flatted_weights = torch.flatten(p)

                # calculando o desvio padrao
                std = torch.std(flatted_weights, -1)

                # calculando o beta para a camada
                pruning_threshold = gamma * std                        
                pruning_thresholds.append(pruning_threshold)

                # mascara booleana, que indica qual peso devera ser mantido
                mask = torch.gt(p.abs(), torch.ones_like(p) * pruning_threshold)
                
                # multiplicando os pesos da camada pelo tensor booleano
                p.multiply_(mask)
                #print(p)
                #p[torch.logical_not(boolean)] = 0

    return pruning_thresholds

# quando varremos os parametros do modelo com model.parameters() temos que verificar o shape do tensor.
# pq pode ser q o tensor seja so um unico valor como um bias por exemplo e nao vai fazer sentido o desvio padrao desse unico valor.
def quantization(model, b, betas=[]):
    # aqui o beta eh usado, pois a quantização usa o mesmo valor em sua formula mesmo. eh pra simplificar a computacao.
    # podemos tambem fazer a funcao pruning_followed_by_quantiztion
    if b >= 1:
        # aqui pode ser uma flag que diz se o pruning foi feito ou não
        if len(betas) > 0:
            with torch.no_grad():
                
                for p, beta in zip(model.parameters(), betas):
                    flatted_weights = torch.flatten(p)
                    qk_prime = (torch.max(torch.abs(flatted_weights)) - beta) / ( (2**(b - 1)) - 1 )
                    torch.round(p/qk_prime, out=p)
                    p.multiply_(qk_prime)

        else:
            with torch.no_grad():
                for p in model.parameters():
                    flatted_weights = torch.flatten(p)
                    beta = 0.0;
                    qk_prime = (torch.max(torch.abs(flatted_weights)) - beta) / ( (2**(b - 1)) - 1 )
                    torch.round(p/qk_prime, out=p)
                    p.multiply_(qk_prime)

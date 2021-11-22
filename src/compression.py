def pruning(model, gamma):
    if gamma > 0.0:
        with torch.no_grad():
            betas = []
            for p in model.parameters():
                # transformando o layer em um vetor unidimensional 
                flatted_weights = torch.flatten(p)

                # calculando o desvio padrao
                std = torch.std(flatted_weights, -1)

                # calculando o beta para a camada
                beta = gamma * std                        
                betas.append(beta)
                
                # mascara booleana, que indica qual peso devera ser mantido
                mask = torch.gt(p.abs(), torch.ones_like(p) * beta)
                
                # multiplicando os pesos da camada pelo tensor booleano
                p.multiply_(mask)
                #print(p)
                #p[torch.logical_not(boolean)] = 0

    return betas

def quantization(model, b, betas=[]):
    if b >= 1:
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

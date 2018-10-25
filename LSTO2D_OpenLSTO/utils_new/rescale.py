import numpy as np

def rescale_lambdas(lambdas, displacements, movelimit):
    maxDisp = 0.
    ndvs = lambdas.shape[0]
    nBpts = displacements.shape[0]
    for dd in range(nBpts):
        disp = abs(displacements[dd])
        if disp > maxDisp: 
            maxDisp = disp
            scale = movelimit / maxDisp

    if maxDisp:
        for pp in range(ndvs):
            lambdas[pp] *= scale
                
    return lambdas
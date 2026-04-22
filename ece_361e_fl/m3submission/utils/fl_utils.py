import torch
import copy


def aggregate_avg(local_weights):
    w_glob = None
    for idx, w_local in enumerate(local_weights):
        if idx == 0:
            w_glob = copy.deepcopy(w_local)
        else:
            for k in w_glob.keys():
                w_glob[k] = torch.add(w_glob[k], w_local[k])

    for k in w_glob.keys():
        w_glob[k] = torch.div(w_glob[k], len(local_weights))

    return w_glob

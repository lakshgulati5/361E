No changes were made to framework code, only config_#.bash files were modified for running different experiments.

M2.2. How do FedMAX and FedProx compare against FedAvg? Which approach is better? Explain.
- It appears that FedMAX and FedProx outperform FedAvg (in terms of accuracy and time to convergence). FedMAX is a better approach then FedProx in terms of the same metrics referred to earlier.

M2.3. FedProx allows for partial computation from each client. In the current implementation, do we
account for hardware heterogeneity using FedProx? If the answer is yes, explain how that works. If the
answer is no, explain what would be needed to make it work
- We don't completely account for hardware heterogeneity with FedProx. This is because we block global aggregation until all edge devices have completed local computation. FedProx does mathematically penalizes local updates that stray too far from the global model with the proximal term "reg_loss".

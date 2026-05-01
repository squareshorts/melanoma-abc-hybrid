import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ABCCalibrator(nn.Module):
    def __init__(self, n_features, eps=1e-4):
        super().__init__()
        self.alpha_0 = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.zeros(n_features))
        self.beta_0 = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
        self.softplus = nn.Softplus()

    def forward(self, s, c):
        a = self.softplus(self.alpha_0 + torch.matmul(c, self.alpha)) + self.eps
        b = self.beta_0 + torch.matmul(c, self.beta)
        logit = a * s + b
        return torch.sigmoid(logit)

def fit_abccalibrator(s_train, c_train, y_train, lambdas=[0, 1e-4, 1e-3, 1e-2, 1e-1]):
    s_t = torch.tensor(s_train, dtype=torch.float32)
    c_t = torch.tensor(c_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    best_loss = float('inf')
    best_model = None
    best_lambda = None
    
    for lam in lambdas:
        torch.manual_seed(42)
        model = ABCCalibrator(c_train.shape[1])
        optimizer = optim.LBFGS(model.parameters(), max_iter=100, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            p = model(s_t, c_t)
            bce = nn.BCELoss()(p, y_t)
            l2 = lam * (torch.norm(model.alpha)**2 + torch.norm(model.beta)**2)
            loss = bce + l2
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        with torch.no_grad():
            p = model(s_t, c_t)
            loss = nn.BCELoss()(p, y_t).item()
        
        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_lambda = lam
            
    print(f"Best lambda chosen: {best_lambda}")
    return best_model

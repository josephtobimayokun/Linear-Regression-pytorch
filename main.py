import torch
from data.synthetic_data import SyntheticRegressionData
from src.linear_regression_model import LinearRegression
from module.trainer import Trainer

data = SyntheticRegressionData(w = torch.tensor([2, -3.14]), b = 4.2)
model = LinearRegression(lr = 0.03, wd = 0.01)
trainer = Trainer(max_epochs = 50)
trainer.fit(data, model)
trainer.plot()
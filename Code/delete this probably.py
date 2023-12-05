# import pandas as pd
# data_file = 'cloud_data.csv'
# df = pd.read_csv(data_file, encoding="utf-8")
# df = df[df['Distance'] <= 100000]
# df.to_csv('distance_cut.csv', index=False)

from torch import randint
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
metric = MulticlassConfusionMatrix(num_classes=5)
metric.update(randint(5, (20,)), randint(5, (20,)))
fig_, ax_ = metric.plot()
plt.show()
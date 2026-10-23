import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_loader as d
from pca import pca

d.main_almoco_sem_nan["Dias Semana"] = d.main_almoco_sem_nan.index.weekday
d.main_almoco_sem_nan["Mês"] = d.main_almoco_sem_nan.index.month
d.main_janta_sem_nan["Mês"] = d.main_janta_sem_nan.index.month

pre_almoco = []
for i in d.main_almoco_sem_nan["Precipitação"]:
    pre_almoco.append(i/1000)
d.main_almoco_sem_nan["preci"] = pre_almoco
pre_janta = []
for i in d.main_janta_sem_nan["Mês"]:
    pre_janta.append(i/1000)
d.main_janta_sem_nan["preci"] = pre_janta

mes_almoco = []
for i in d.main_almoco_sem_nan["Mês"]:
    mes_almoco.append(d.valor_mes[i - 1])
d.main_almoco_sem_nan["Valor Mês"] = mes_almoco
mes_janta = []
for i in d.main_janta_sem_nan["Mês"]:
    mes_janta.append(d.valor_mes[i - 1])
d.main_janta_sem_nan["Valor Mês"] = mes_janta

valor_categoria = []
for categoria in d.main_almoco_sem_nan["Categoria do Prato"]:
  valor_categoria.append(int(d.almoco_categorias_ordenadas.index.get_loc(categoria)))
d.main_almoco_sem_nan["Valor Categoria"] = valor_categoria

result_almoco = []
for j in range(len(d.main_almoco_sem_nan.index)):
    result = []
    for i in range(10):
            if i == int(np.floor(d.main_almoco_sem_nan["Total"].iloc[j]/1000)):
                result.append(1)
            else:
                result.append(0)
    result_almoco.append(int(np.argmax(result)))
result_janta = []
for j in range(len(d.main_janta_sem_nan.index)):
    result = []
    for i in range(10):
            if i == int(np.floor(d.main_janta_sem_nan["Total"].iloc[j]/1000)):
                result.append(1)
            else:
                result.append(0)
    result_janta.append(int(np.argmax(result)))

d.main_almoco_sem_nan["target"] = result_almoco
d.main_janta_sem_nan["target"] = result_janta
pca_table_a = d.main_almoco_sem_nan.copy()
pca_table_j = d.main_janta_sem_nan.copy()
new_a = pca_table_a.drop(columns=["target", "Valor Categoria", "Precipitação", "Categoria do Prato", "Prato Principal", "RU", "RS", "RA", "Total", "Dia da Semana", "Mês", "Dias Semana"])
new_j = pca_table_j.drop(columns=["target", "Categoria do Prato", "Precipitação", "Prato Principal", "RU", "RS", "RA", "Total", "Dia da Semana", "Mês"])
new_index_a = d.main_almoco_sem_nan["target"]
new_index_j = d.main_janta_sem_nan["target"]
new_a = new_a.set_index(new_index_a)
new_j = new_j.set_index(new_index_j)

# Initialize to reduce the data up to the nubmer of componentes that explains 95% of the variance.
model_a = pca(n_components=0.80)
model_j = pca(n_components=0.80)

# Reduce the data towards 3 PCs
#model_a = pca(n_components=3)
#model_j = pca(n_components=3)

# Fit transform
results_a = model_a.fit_transform(new_a)
results_j = model_j.fit_transform(new_j)

print(model_a.results['explained_var'])
print(model_j.results['explained_var'])
print(model_a.results['variance_ratio'])
print(model_j.results['variance_ratio'])

# Make plot
fig_a, ax_a = model_a.plot()
fig_j, ax_j = model_j.plot()

ax_a3 = model_a.biplot3d(n_feat=10, legend=False)
ax_j3 = model_j.biplot(n_feat=10, legend=False)
plt.show()

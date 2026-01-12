import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein as lev
from scipy.stats import ttest_ind
import unicodedata
import datetime

# Dados de Volumetria do Almoço

df_volumetria_almoco = pd.read_csv(r'C:\Users\João Pedro\Pessoal\Estudos\MS571\data\volumetria_almoco.csv')

def try_parsing_datetime(x):
    fmt = '%Y-%m-%d'
    try:
        return pd.to_datetime(x, format=fmt)
    except (ValueError, TypeError):
        return pd.NaT
df_volumetria_almoco['Data'] = df_volumetria_almoco['Data'].apply(try_parsing_datetime)


# Dados de Volumetria da Janta

df_volumetria_janta = pd.read_csv(r'C:\Users\João Pedro\Pessoal\Estudos\MS571\data\volumetria_janta.csv')

df_volumetria_janta['Data'] = df_volumetria_janta['Data'].apply(try_parsing_datetime)


# Dados do Cardápio

df_cardapio = pd.read_csv(r'C:\Users\João Pedro\Pessoal\Estudos\MS571\data\cardapio.csv')


df_cardapio['date'] = df_cardapio['date'].apply(try_parsing_datetime)

df_cardapio_almoco_padrao = df_cardapio.loc[df_cardapio['refeicao'] == "Almoço - Cardápio Padrão"]

df_cardapio_janta_padrao = df_cardapio.loc[df_cardapio['refeicao'] == "Jantar - Cardápio Padrão"]


# Dados de Precipitação 2024

df_precipitacao_2024 = pd.read_csv(r'C:\Users\João Pedro\Pessoal\Estudos\MS571\data\precip_horarios_2024_Cepagri.csv')


prec_por_refeicao_2024 = df_precipitacao_2024["Precipitação"].groupby(df_precipitacao_2024.index // 12).sum()

df_prec_por_refeicao_2024 = prec_por_refeicao_2024.to_frame()


df_prec_por_refeicao_2024["Refeição"] = df_prec_por_refeicao_2024.index.map(lambda x: "Almoço" if x % 2 == 0 else "Jantar")


df_prec_almoco_2024 = df_prec_por_refeicao_2024.loc[df_prec_por_refeicao_2024['Refeição'] == "Almoço"]


df_prec_janta_2024 = df_prec_por_refeicao_2024.loc[df_prec_por_refeicao_2024['Refeição'] == "Jantar"]


# Dados de Precipitação 2025


df_precipitacao_2025 = pd.read_csv(r'C:\Users\João Pedro\Pessoal\Estudos\MS571\data\precip_horarios_2025parcial_Cepagri.csv')


prec_por_refeicao_2025 = df_precipitacao_2025["Precipitação"].groupby(df_precipitacao_2025.index // 12).sum()

df_prec_por_refeicao_2025 = prec_por_refeicao_2025.to_frame()


df_prec_por_refeicao_2025["Refeição"] = df_prec_por_refeicao_2025.index.map(lambda x: "Almoço" if x % 2 == 0 else "Jantar")


df_prec_almoco_2025 = df_prec_por_refeicao_2025.loc[df_prec_por_refeicao_2025['Refeição'] == "Almoço"]

df_prec_janta_2025 = df_prec_por_refeicao_2025.loc[df_prec_por_refeicao_2025['Refeição'] == "Jantar"]


# Dados Precipitação 2024 + 2025

df_prec_almoco_2024_2025 = pd.concat([df_prec_almoco_2024, df_prec_almoco_2025])


df_prec_janta_2024_2025 = pd.concat([df_prec_janta_2024, df_prec_janta_2025])


df_prec_janta_2024_2025 = df_prec_janta_2024_2025.reset_index(drop=True)

df_prec_almoco_2024_2025 = df_prec_almoco_2024_2025.reset_index(drop=True)

# Main Table

#Criando a main do almoço e da janta a partir da tabela da volumetria

main_almoco_raw = df_volumetria_almoco

main_janta_raw = df_volumetria_janta

main_almoco_raw["Precipitação"] = df_prec_almoco_2024_2025["Precipitação"].reindex(main_almoco_raw.index)

main_janta_raw["Precipitação"] = df_prec_janta_2024_2025["Precipitação"].reindex(main_janta_raw.index)

main_almoco_raw = main_almoco_raw.set_index("Data")

main_janta_raw = main_janta_raw.set_index("Data")

df_cardapio_almoco_padrao = df_cardapio_almoco_padrao.set_index("date")

df_cardapio_janta_padrao = df_cardapio_janta_padrao.set_index("date")

main_almoco_raw = main_almoco_raw.join(df_cardapio_almoco_padrao["prato"], how="left")

main_janta_raw = main_janta_raw.join(df_cardapio_janta_padrao["prato"], how="left")

main_janta_raw = main_janta_raw.rename(columns={"prato": "Prato Principal"})


main_almoco_raw = main_almoco_raw.rename(columns={"prato": "Prato Principal"})


# Produto Final

main_almoco_sem_nan = main_almoco_raw.dropna().copy()

main_janta_sem_nan = main_janta_raw.dropna().copy()


categorias = {}

def tirar_acento(texto):
    texto_normalizado = unicodedata.normalize('NFKD', texto)
    texto_sem_acentos = ''.join(c for c in texto_normalizado if not unicodedata.combining(c))
    return texto_sem_acentos

def limpar(descricao):
# tirar tudo depois de parenteses
    descricao = descricao.split('(')[0]
    descricao = descricao.lower()
    descricao = tirar_acento(descricao)
    descricao_palavras = descricao.split()
    palavras_reais = []
    for palavra in descricao_palavras:
        if len(palavras_reais) == 2:
            break
        if len(palavra) > 3:
            palavras_reais.append(palavra)
    descricao = ' '.join(palavras_reais)
    return descricao

def categorizar(descricao):
    threshold = 0.8
    descricao_limpa = limpar(descricao)
    for categoria in categorias.keys():
        if lev.ratio(descricao_limpa, categoria) > threshold:
            categorias[categoria].append(descricao)
            return categoria
    categorias[descricao_limpa] = [descricao]
    return descricao_limpa


main_almoco_sem_nan["Categoria do Prato"] = main_almoco_sem_nan["Prato Principal"].apply(categorizar)
main_janta_sem_nan["Categoria do Prato"] = main_janta_sem_nan["Prato Principal"].apply(categorizar)

almoco_categorias = main_almoco_sem_nan.groupby("Categoria do Prato")["Total"].mean()


almoco_categorias_ordenadas = almoco_categorias.sort_values(ascending=True)


janta_categorias = main_janta_sem_nan.groupby("Categoria do Prato")["Total"].mean()

janta_categorias_ordenadas = janta_categorias.sort_values(ascending=True)

metrica_almoco = []
metrica_janta = []
for categoria in almoco_categorias.index:
    metrica_almoco.append([categoria, int(almoco_categorias['{}'.format(categoria)])/1000])
for categoria in janta_categorias.index:
    metrica_janta.append([categoria, int(janta_categorias['{}'.format(categoria)])/1000])


main_almoco_sem_nan["Dia da Semana"] = [main_almoco_sem_nan.index[i].strftime("%A") for i in range(len(main_almoco_sem_nan.index))]
main_janta_sem_nan["Dia da Semana"] = [main_janta_sem_nan.index[i].strftime("%A") for i in range(len(main_janta_sem_nan.index))]

Valor_Dia_Semana = []
for j in range(len(main_almoco_sem_nan.index)):     
    if main_almoco_sem_nan["Dia da Semana"].iloc[j] == "Friday":
        Valor_Dia_Semana.append(3)
    elif main_almoco_sem_nan["Dia da Semana"].iloc[j] == "Saturday":
        Valor_Dia_Semana.append(2)
    elif main_almoco_sem_nan["Dia da Semana"].iloc[j] == "Sunday":
        Valor_Dia_Semana.append(1)
    else:
        Valor_Dia_Semana.append(5)
main_almoco_sem_nan["Valor Dia da Semana"] = Valor_Dia_Semana
main_janta_sem_nan["Valor Dia da Semana"] = Valor_Dia_Semana

Valor_Cardapio = []
for j in range(len(main_almoco_sem_nan.index)):
    if main_almoco_sem_nan["Categoria do Prato"].iloc[j] == "peixe":
        Valor_Cardapio.append(1)
    else:
        Valor_Cardapio.append(3)
main_almoco_sem_nan["Valor Cardapio"] = Valor_Cardapio

main_almoco_sem_nan = main_almoco_sem_nan.sample(frac=1)
main_janta_sem_nan = main_janta_sem_nan.sample(frac=1)


valor_mes = [1,1,4,3,3,3,1,3,3,3,3,1]

drop = []
for i in range(len(main_almoco_sem_nan.index)):
    if main_almoco_sem_nan['RS'].iloc[i] == 0:
        drop.append(main_almoco_sem_nan.index[i])
main_almoco_sem_nan = main_almoco_sem_nan.drop(drop)

drop = []
for i in range(len(main_janta_sem_nan.index)):
    if main_janta_sem_nan['RS'].iloc[i] == 0:
        drop.append(main_janta_sem_nan.index[i])
main_janta_sem_nan = main_janta_sem_nan.drop(drop)


def load_data(restaurante):
    dados_treino_almoço = main_almoco_sem_nan[0:278]
    dados_treino_janta = main_janta_sem_nan[0:278]
    dados_test_almoço = main_almoco_sem_nan[279:329]
    dados_test_janta = main_janta_sem_nan[279:329]
    training_data_almoco = []
    training_data_janta = []
    test_data_almoco = []
    test_data_janta = []
    result_training_almoco = []
    result_training_janta = []
    result_test_almoco = []
    result_test_janta = []
    nota_training_almoco = []
    nota_training_janta = []
    nota_test_almoco = []
    nota_test_janta = []
    for j in range(len(dados_treino_almoço.index)):
        result = []
        for i in range(10):
            if restaurante == "RU":
                if i == int(np.floor(dados_treino_almoço["{}".format(restaurante)].iloc[j]/500)):
                    result.append(1)
                else:
                    result.append(0)
            else:
                if i == int(np.floor(dados_treino_almoço["{}".format(restaurante)].iloc[j]/200)):
                    result.append(1)
                else:
                    result.append(0)
        result_training_almoco.append(result)
        nota_training_almoco.append((int(almoco_categorias[dados_treino_almoço['Categoria do Prato'].iloc[j]])/1000)**2)
    for j in range(len(dados_treino_janta.index)):
        result = []
        for i in range(10):
            if restaurante == "RU":
                if i == int(np.floor(dados_treino_janta["{}".format(restaurante)].iloc[j]/500)):
                    result.append(1)
                else:
                    result.append(0)
            else:
                if i == int(np.floor(dados_treino_janta["{}".format(restaurante)].iloc[j]/200)):
                    result.append(1)
                else:
                    result.append(0)
        result_training_janta.append(result)
        nota_training_janta.append((int(janta_categorias[dados_treino_janta['Categoria do Prato'].iloc[j]])/1000)**2)
    
    for j in range(len(dados_test_almoço.index)):
        result = []
        for i in range(10):
            if restaurante == "RU":
                if i == int(np.floor(dados_test_almoço["{}".format(restaurante)].iloc[j]/500)):
                    result.append(1)
                else:
                    result.append(0)
            else:
                if i == int(np.floor(dados_test_almoço["{}".format(restaurante)].iloc[j]/200)):
                    result.append(1)
                else:
                    result.append(0)
        result_test_almoco.append(result)
        nota_test_almoco.append((int(almoco_categorias[dados_test_almoço['Categoria do Prato'].iloc[j]])/1000)**2)
    for j in range(len(dados_test_janta.index)):
        result = []
        for i in range(10):
            if restaurante == "RU":
                if i == int(np.floor(dados_test_janta["{}".format(restaurante)].iloc[j]/500)):
                    result.append(1)
                else:
                    result.append(0)
            else:
                if i == int(np.floor(dados_test_janta["{}".format(restaurante)].iloc[j]/200)):
                    result.append(1)
                else:
                    result.append(0)
        result_test_janta.append(result)
        nota_test_janta.append((int(janta_categorias[dados_test_janta['Categoria do Prato'].iloc[j]])/1000)**2)

    # Dados de treino e teste pro almoço
    for i in range(len(dados_treino_almoço.index)):
        training_data_almoco.append([[int(dados_treino_almoço["Valor Dia da Semana"].iloc[i]), valor_mes[dados_treino_almoço.index[i].month - 1], int(dados_treino_almoço["Precipitação"].iloc[i])/1000, int(dados_treino_almoço["Valor Cardapio"].iloc[i])], result_training_almoco[i]])
    for i in range(len(dados_test_almoço.index)):
        test_data_almoco.append([[int(dados_test_almoço["Valor Dia da Semana"].iloc[i]), valor_mes[dados_test_almoço.index[i].month - 1], int(dados_test_almoço["Precipitação"].iloc[i])/1000, int(dados_test_almoço["Valor Cardapio"].iloc[i])], result_test_almoco[i]])
    #Dados de treino e teste pra janta
    for i in range(len(dados_treino_janta.index)):
        training_data_janta.append([[int(dados_treino_janta["Valor Dia da Semana"].iloc[i]), valor_mes[dados_treino_janta.index[i].month - 1], int(dados_treino_janta["Precipitação"].iloc[i])/1000, nota_training_janta[i]], result_training_janta[i]])
    for i in range(len(dados_test_janta.index)):
        test_data_janta.append([[int(dados_test_janta["Valor Dia da Semana"].iloc[i]), valor_mes[dados_test_janta.index[i].month - 1], int(dados_test_janta["Precipitação"].iloc[i])/1000, nota_test_janta[i]], result_test_janta[i]])

    return (training_data_almoco, test_data_almoco, training_data_janta, test_data_janta)



"""plt.figure(figsize=(8,5))
plt.hist(main_almoco_sem_nan["RU"], bins=50, edgecolor='black')
plt.title("Histograma de Volumetria de Refeições no RU")
plt.figure(figsize=(8,5))
plt.hist(main_almoco_sem_nan["RS"], bins=30, edgecolor='black')
plt.title("Histograma de Volumetria de Refeições no RS")
plt.figure(figsize=(8,5))
plt.hist(main_almoco_sem_nan["RA"], bins=30, edgecolor='black')
plt.title("Histograma de Volumetria de Refeições no RA")
plt.show()"""


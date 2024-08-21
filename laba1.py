import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.spatial
import math


def manhattan(a, b):
    distance = 0
    for i in range(len(a)):
        distance += abs(a[i] - b[i])
    return distance


def euclidean(a, b):
    distance = 0
    for i in range(len(a)):
        distance += pow(a[i] - b[i], 2)
    return math.sqrt(distance)


def cosine(a, b):
    distance = scipy.spatial.distance.cosine(a, b)
    return distance


laptopTree = [["Встроенная", "Дискретная"],
              ["Высокая", "Низкая"],
              ["Ноутбуки", "Ультрабуки", "Геймерские"],
              [4, 8, 16, 32],
              ["AMD", "Intel"]]

weights = [0.4, 0.3, 0.2, 0.3, 0.1]


def diff_tree(t1, t2):
    difftree = []
    for i in range(0, 5):
        difftree.append(abs(laptopTree[i].index(t1[i]) - laptopTree[i].index(t2[i])))
    similarity = 0
    for i in range(len(difftree)):
        similarity += difftree[i] * weights[i]
    return similarity


def printCorrelationMatrix(ds, metric):
    matr = []
    for i in range(len(ds.values.tolist())):
        r = []
        for k in range(len(ds.values.tolist())):
            r.append(metric(ds.values.tolist()[i], ds.values.tolist()[k]))
        matr.append(np.array(r))
    matr = np.array(matr)

    plt.imshow(matr)
    plt.title(metric)
    plt.xticks(np.arange(0, len(ds.values.tolist())), rotation=-45)
    plt.yticks(np.arange(0, len(ds.values.tolist())))
    figure(figsize=(10, 10), dpi=160)
    plt.show()


dataSetFromTxt = pd.read_csv('laptops.txt', delimiter='\t', encoding="utf-16-le")
ds = dataSetFromTxt.copy(deep=True)
del ds["Ноутбук"]
del ds["Теги"]

ds["Тип видеокарты"], _ = pd.factorize(ds["Тип видеокарты"])
ds["Цена"], _ = pd.factorize(ds["Цена"])
ds["Категория"], _ = pd.factorize(ds["Категория"])
ds["DDR4"], _ = pd.factorize(ds["DDR4"])
ds["ЗУ"], _ = pd.factorize(ds["ЗУ"])
ds["Количество_ОЗУ"] = ds["Количество_ОЗУ"].values / max(ds["Количество_ОЗУ"].values)
ds["Диагональ больше 14?"], _ = pd.factorize(ds["Диагональ больше 14?"])
ds["Количество на складе"] = ds["Количество на складе"].values / max(ds["Количество на складе"].values)
ds["Есть подсветка клавиатуры"], _ = pd.factorize(ds["Есть подсветка клавиатуры"])
ds["Есть отпечаток пальца"], _ = pd.factorize(ds["Есть отпечаток пальца"])
ds["Процессор"], _ = pd.factorize(ds["Процессор"])
ds["Есть Ethernet"], _ = pd.factorize(ds["Есть Ethernet"])
ds["Объем ЗУ"] = ds["Объем ЗУ"].values / max(ds["Объем ЗУ"].values)
ds["Цвет"], _ = pd.factorize(ds["Цвет"])


# print(ds)


# printCorrelationMatrix(ds, manhattan)
#
# printCorrelationMatrix(ds, euclidean)
#
# printCorrelationMatrix(ds, cosine)
#
# printCorrelationMatrix(dataSetFromTxt, diff_tree)


def getSimilarsByLaptopSerialNumber(ds, dataSet, metric, serial_number):
    r = []
    for i in range(len(ds.values.tolist())):
        r.append(metric(ds.values.tolist()[serial_number], ds.values.tolist()[i]))

    return pd.DataFrame(list(zip(r, map(lambda e: str("   ".join(e[-1:])),
                                        dataSet.values.tolist()))), index=np.arange(len(r)),
                        columns=['Величина различия', 'Ноут'])


# Поиск рекомендаций для одного ноутбука. Начальные параметры:
serial_number = 13
distance_calculation = 'manhattan'
# distance_calculation = 'euclidean'
# distance_calculation = 'diff_tree'
# distance_calculation = 'cosine'

# print("\n\nИсходный ноутбук: ", dataSetFromTxt.values.tolist()[serial_number])
# print("Поиск... ")

func = "diff_tree"
dataset = ""

if distance_calculation == "manhattan":
    func = manhattan
    dataset = ds
elif distance_calculation == "euclidean":
    func = euclidean
    dataset = ds
elif distance_calculation == "diff_tree":
    func = diff_tree
    dataset = dataSetFromTxt
elif distance_calculation == "cosine":
    func = cosine
    dataset = ds


# print("Похожие товары:\n",
#       getSimilarsByLaptopSerialNumber(dataset, dataSetFromTxt, func, serial_number).sort_values("Величина различия"))
#
# plt.plot(serial_number, serial_number, 'r*')
# printCorrelationMatrix(dataset, func)

def getSimilarsByGroupLaptops(ds, dataSetFromTxt, metric, like_serial_number, dislikes):
    likeVec = []
    if (len(like_serial_number) > 0):
        for k in like_serial_number:
            likeVec.append(
                np.array(getSimilarsByLaptopSerialNumber(ds, dataSetFromTxt, metric, k)["Величина различия"]))

    mostRelated = pd.DataFrame()
    r = []
    for k in range(len(ds.values.tolist())):
        if len(like_serial_number) > 0:
            tt = np.sum([np.array(getSimilarsByLaptopSerialNumber(ds, dataSetFromTxt, metric, k)["Величина различия"]),
                         np.average(likeVec, 0)], 0)

        mostRelated = mostRelated.append(
            {"id": np.argmin(tt),
             "Характеристики": " ".join(list(map(str, dataSetFromTxt.values.tolist()[np.argmin(tt)]))[:5]),
             "Разница": np.amin(tt)}, ignore_index=True)
        r.append(tt)

    # print(r)
    matr = r
    plt.imshow(matr)
    plt.xticks(np.arange(0, len(ds.values.tolist())), rotation=-45)
    plt.yticks(np.arange(0, len(ds.values.tolist())))
    figure(figsize=(10, 10), dpi=160)
    plt.show()
    mostRelated = mostRelated.drop_duplicates(subset='id', keep="last")
    for k in like_serial_number:
        mostRelated = mostRelated.drop(k)

    for k in dislikes:
        mostRelated = mostRelated.drop(k)
    mostRelated = mostRelated.sort_values('Разница')
    print(mostRelated)

    return r


# Поиск рекомендаций для группы ноутбуков. Начальные параметры:
# favorites = "27,26,25"
# dislikes = "6,7"
# getSimilarsByGroupLaptops(ds, dataSetFromTxt, func, np.fromstring(favorites, dtype=int, sep=','),
#                           np.fromstring(dislikes, dtype=int, sep=','))


# Параметрический поиск
prices = ['высокая', 'низкая']
gpu = ['встроенная', 'дискретная']
laptop_types = ['ноутбуки', 'ультрабуки', 'геймерские']
ram_amounts = [4, 8, 16, 32]
cpu = ['amd', 'intel']

while True:
    gpu_type = str(input("Введите тип желаемой видеокарты (Встроенная или Дискретная): "))
    if gpu_type.lower() in gpu:
        break
    print("Вы ввели некорректные данные")

while True:
    prices_category = str(input("Введите желаемую ценовую категорию (Высокая или Низкая): "))
    if prices_category.lower() in prices:
        break
    print("Вы ввели некорректные данные")

while True:
    laptop_type = str(input("Выберите желаемую категорию (Ноутбуки, Ультрабуки или Геймерские): "))
    if laptop_type.lower() in laptop_types:
        break
    print("Вы ввели некорректные данные")

while True:
    ram_amount = int(input("Введите желаемое количество оперативной памяти (4, 8, 16, 32): "))
    if ram_amount in ram_amounts:
        break
    print("Вы ввели некорректные данные")

while True:
    cpu_type = str(input("Введите желаемый процессор (AMD, Intel): "))
    if cpu_type.lower() in cpu:
        break
    print("Вы ввели некорректные данные")

if cpu_type == "amd":
    cpu_type = cpu_type.upper()
else:
    cpu_type = cpu_type.title()

dataSetSearch = pd.read_csv('laptops.txt', delimiter='\t', encoding="utf-16-le",
                            usecols=['Тип видеокарты', 'Цена', 'Категория', 'Количество_ОЗУ', 'Процессор', 'Ноутбук'])
# dataSetSearch = pd.read_csv('laptops.txt', delimiter='\t', encoding="utf-16-le")

gpu_search = dataSetSearch[dataSetSearch["Тип видеокарты"] == gpu_type.title()]
prices_search = gpu_search[gpu_search["Цена"] == prices_category.title()]
laptop_type_search = prices_search[prices_search["Категория"] == laptop_type.title()]
ram_amounts_search = laptop_type_search[laptop_type_search["Количество_ОЗУ"] == ram_amount]
cpu_search = ram_amounts_search[ram_amounts_search["Процессор"] == cpu_type]
print("\n")


def getSimilarsInSearch(ds, dataSet, metric, dfSearch):
    r = []
    for i in range(len(ds.values.tolist())):
        r.append(metric(dfSearch.values.tolist()[0], ds.values.tolist()[i]))

    return pd.DataFrame(list(zip(r, map(lambda e: str("   ".join(e[-1:])),
                                        dataSet.values.tolist()))), index=np.arange(len(r)),
                        columns=['Величина различия', 'Ноут'])


def searchSimulariry():
    dfSearch = pd.DataFrame({'Тип видеокарты': [gpu_type.title()],
                             'Цена': [prices_category.title()],
                             'Категория': [laptop_type.title()],
                             'Количество_ОЗУ': ram_amount,
                             'Процессор': cpu_type})

    func = diff_tree
    dataset = dataSetSearch
    print(getSimilarsInSearch(dataset, dataSetSearch, func, dfSearch).sort_values("Величина различия"))


if cpu_search.empty:
    print("В наличии нет ноутбуков с такими параметрами. Возможно вас заинтересуют похожие ноутбуки: ")
    searchSimulariry()
else:
    print("Результаты поиска: ")
    print(cpu_search)

# n = num of features
# m = num of examples

import csv
import math

examples_raw = []
Y = []

def calc_ticket_frequency(ticket):
    count = 0
    for x in examples_raw:
        ticket_x = x[6]
        if ticket == ticket_x:
            count += 1
    return count

def is_in_cabin(str, c):
    try:
        str.index(c)
        return True
    except:
        return False

def create_feature_vector(pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked):
    return [
        1,
        1 if pclass == 1 else 0,
        1 if pclass == 2 else 0,
        1 if pclass == 3 else 0,
        1 if sex == "male" else 0,
        1 if sex == "female" else 0,
        age,
        sibsp,
        parch,
        calc_ticket_frequency(ticket),
        fare,
        len(cabin) > 0, # has cabin
        is_in_cabin(cabin, "A"),
        is_in_cabin(cabin, "B"),
        is_in_cabin(cabin, "C"),
        is_in_cabin(cabin, "D"),
        is_in_cabin(cabin, "E"),
        is_in_cabin(cabin, "F"),
        is_in_cabin(cabin, "G"),
        1 if embarked == "C" else 0,
        1 if embarked == "Q" else 0,
        1 if embarked == "S" else 0,
    ]


with open("train.csv", 'r') as file:
    csvreader = csv.reader(file)
    headers = next(csvreader)
    for row in csvreader:
        survived = int(row[1])
        pclass = int(row[2])
        sex = row[4]
        age = float(row[5]) if len(row[5]) > 0 else -1
        sibsp = int(row[6])
        parch = int(row[7])
        ticket = row[8]
        fare = float(row[9])
        cabin = row[10]
        embarked = row[11]

        if age == -1:
            continue

        raw = [pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked]
        examples_raw.append(raw)
        Y.append(survived)

examples = []

for raw in examples_raw:
    x = create_feature_vector(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8])
    examples.append(x)

n = 21
m = len(examples)

print("m examples :", m)

def dot_product(a, b):
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

def z(theta, x):
    return dot_product(theta, x)

def h(theta, x):
    return 1 / (1 + math.exp(-1 * z(theta, x)))

ALPHA = 0.01

def gradient_ascent(theta):
    new_theta = list(theta)

    for i in range(m):
        y_i = Y[i]
        x_i = examples[i]

        diff = y_i - h(theta, x_i)

        for j in range(n + 1):
            new_theta[j] += diff * x_i[j] * ALPHA / m

    return new_theta

def calc_avg_likelihood(theta):
    sum = 0
    for i in range(m):
        x_i = examples[i]
        y_i = Y[i]
        hypothesis = h(theta, x_i)
        sum += math.pow(hypothesis, y_i) * math.pow(1 - hypothesis, 1 - y_i)
    return sum / m

EPOCHS = 10000
theta = [0] * (n + 1)
for e in range(EPOCHS):
    if e % 100 == 0:
        avg_loss = calc_avg_likelihood(theta)
        print("epoch :", e)
        print("likelihood :", avg_loss)

    theta = gradient_ascent(theta)

print("done after epochs :", EPOCHS)
print("likelihood :", calc_avg_likelihood(theta))

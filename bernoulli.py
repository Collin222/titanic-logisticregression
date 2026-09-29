# n = num of features
# m = num of examples
# a GLM using a bernoulli PDF

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

def h(theta, x):
    eta = dot_product(theta, x)
    return 1 / (1 + math.exp(-1 * eta))

ALPHA = 0.01

def apply_gradient(theta):
    new_theta = list(theta)

    # batch gradient ascent
    # maximize likelihood
    for i in range(m):
        x_i = examples[i]
        y_i = Y[i]

        prediction = h(theta, x_i)
        error = y_i - prediction
        for j in range(n + 1):
            grad = error * x_i[j]
            new_theta[j] += grad / m * ALPHA

    return new_theta

def calc_avg_likelihood(theta):
    sum = 0
    for i in range(m):
        x_i = examples[i]
        y_i = Y[i]
        hypothesis = h(theta, x_i)
        sum += math.pow(hypothesis, y_i) * math.pow(1 - hypothesis, 1 - y_i)
    return sum / m

def calc_accuracy(theta):
    correct = 0
    for i in range(m):
        pred = 1 if h(theta, examples[i]) >= 0.5 else 0
        if pred == Y[i]:
            correct += 1
    return correct / m

EPOCHS = 10000
def train():
    theta = [0] * (n + 1)

    for e in range(EPOCHS):
        if e % 100 == 0:
            likelihood = calc_avg_likelihood(theta)
            accuracy = calc_accuracy(theta)
            print("epoch", e)
            print("avg likelihood", likelihood)
            print("accuracy", accuracy)

        theta = apply_gradient(theta)
    
train()
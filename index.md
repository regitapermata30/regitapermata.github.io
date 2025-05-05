---
title: "Regresi Logistik"
revealOptions:
  transition: 'slide'
---

## 🤖 Regresi Logistik

Regresi logistik digunakan untuk klasifikasi, bukan regresi nilai kontinu.

---

## 📐 Fungsi Sigmoid

\[
P(y = 1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
\]

---

## 🧠 Aturan Klasifikasi

Jika:

- \( P(y = 1) > 0.5 \) → kelas = 1  
- \( P(y = 1) \leq 0.5 \) → kelas = 0

---

## 📊 Visualisasi Sigmoid

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.linspace(-10, 10, 200)
y = sigmoid(x)

plt.plot(x, y)

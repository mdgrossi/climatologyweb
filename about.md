---
layout: post
title: Equations
---

```
Let $\psi(t) = \set{\psi_i; i=1,2,...} = \mathbb{T}$ be an ongoing observational time series of environmental parameter $\psi$ and define the following subsets:
- $\mathbb{S_s} \subseteq \mathbb{T}$ such that $\mathbb{S_s}$ contains only those measurements recorded during day $s$
- $\mathbb{D_d} \subseteq \mathbb{T}$ such that $\mathbb{D_d}$ contains only those measurements recorded during day-of-year $d$ (*e.g.*, $d = 1 =$ January 1,..., $d = 366 =$ December 31)
- $\mathbb{M_m} \subseteq \mathbb{T}$ such that $\mathbb{M_m}$ contains only those measurements recorded during month $m$
- $\mathbb{Y_y} \subseteq \mathbb{T}$ such that $\mathbb{Y_y}$ contains only those measurements recorded during year $y$
```

Let $\psi(t) = \set{\psi_i; i=1,2,...} = \mathbb{T}$ be an ongoing observational time series of environmental parameter $\psi$ and define the following matrices:
- Let $\mathbb{S}_{d,y}$ be $\mathbb{T}$ arranged by day $s$ and year $y$
- Let $\mathbb{D}_{d,y}$ be $\mathbb{T}$ arranged by day-of-year $d$ and year $y$

## Daily Calculations

### Daily High

$\mathbb{H_s} = \set{\max\set{\psi\_i}_{i \in \mathbb{S_s}}}$

where $\mathbb{H}$ is the set of high values for all days in $\mathbb{T}$.

---

### Daily Low

$\mathbb{L_s} = \set{\min\set{\psi\_i}_{i \in \mathbb{S_s}}}$

where $\mathbb{L}$ is the set of low values for all days in $\mathbb{T}$.

---

### Daily Average

$\mathbb{A}_s = \set{\frac{1}{2} \big( \mathbb{H_s} + \mathbb{L_s} \big)}$

where $\mathbb{A}$ is the set of averages for all days in $\mathbb{T}$.

---

### Record High Daily Average

$\mathbb{A}\_\text{hi}^{[d]} = \max\set{ \mathbb{A}^{[d]}\_{i} }_{i \in \mathbb{D_d}} $

---

### Record Low Daily Average

$\mu_\text{hi}^{[s]} = \min\set{ \mu^{[s]}\_{i} }_{i \in \mathbb{S^{[s]}}} $

---

### Daily Average High

$\mathbb{H}^{[s]} = \max\set{y\_i}_{i \in \mathbb{S^{[s]}}}$

$\frac{1}{N}\sum\limits_{n=1}^{N}$

Let $s \subseteq d$:

$\frac{1}{\mathbb{N}}$ $\sum\limits_{n}^{\mathbb{N}} \max(y^{[s]}) $

$\frac{\sum\limits_{n=1}^{d} y_\text{max}}{d}$

---

### Daily Record High

Let $s \subseteq d$:

$\frac{1}{\mathbb{N}}$ $\sum\limits_{n}^{\mathbb{N}} \max(y^{[s]}) $

$\frac{\sum\limits_{n=1}^{d} y_\text{max}}{d}$

---

### Daily Record Low

Let $s \subseteq d$:

$\frac{1}{\mathbb{N}}$ $\sum\limits_{n}^{\mathbb{N}} \max(y^{[s]}) $

$\frac{\sum\limits_{n=1}^{d} y_\text{max}}{d}$

--

## Monthly Calculations

### Monthly Average

$\overline{y\_\text{mon}} = \frac{1}{d} \sum\limits_{n=1}^{d} \overline{y}_\text{day}^{n}$

where $d =$ number of days in the month


### Record High Monthly Average

Maximum $\overline{y\_\text{mon}}$ for the given month over the course of the observational time series

### Record Low Monthly Average

Minimum $\overline{y\_\text{mon}}$ for the given month over the course of the observational time series



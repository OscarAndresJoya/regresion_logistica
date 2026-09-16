"""
===============================================================================
Introducción al Machine Learning
Taller en clase: Modelo logístico y máxima verosimilitud
===============================================================================

Este script recorre los tres ejercicios del taller en orden. La idea de fondo
que conecta a los tres es la misma: cómo encontrar los parámetros theta que
mejor explican la relación entre x y la probabilidad de que y = 1.

  - Ejercicio 1: se busca el mejor theta por FUERZA BRUTA, comparando la
    verosimilitud de una lista finita de candidatos.
  - Ejercicio 2: se busca el mejor theta por OPTIMIZACIÓN NUMÉRICA continua,
    minimizando el riesgo empírico con la pérdida logística.
  - Ejercicio 3: se muestra que el modelo logístico es una REGRESIÓN LINEAL
    en el espacio de los logits, y se recupera theta por esa vía.

Las funciones auxiliares (sigmoid, compute_probs, likelihood, softmax,
logistic_loss, empirical_risk, logit) viven en src/functions.py.
===============================================================================
"""

from src import functions as fc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =============================================================================
# DATOS DEL TALLER
# Se definen una sola vez al inicio, porque los tres ejercicios usan el mismo
# conjunto de datos. Son 9 observaciones: las primeras 4 tienen etiqueta 0 y
# las últimas 5 tienen etiqueta 1, con el cambio ocurriendo en x = 0.
# =============================================================================
x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])


# =============================================================================
# EJERCICIO 1 — Máxima verosimilitud por búsqueda en una lista finita
# =============================================================================

# -----------------------------------------------------------------------------
# Ejercicio 1.1 — Función logística (sigmoide)
# Verificamos que sigmoid(t) = 1 / (1 + e^(-t)) se comporta como esperamos:
# vale 0.5 en t = 0 y satura hacia 1 y hacia 0 en los extremos.
# -----------------------------------------------------------------------------
print("=== Ejercicio 1.1: prueba de la sigmoide ===")
print(fc.sigmoid(0))      # debería dar 0.5
print(fc.sigmoid(100))    # debería acercarse a 1
print(fc.sigmoid(-100))   # debería acercarse a 0


# -----------------------------------------------------------------------------
# Ejercicio 1.2 — Probabilidades p_i = sigma(theta0 + theta1 * x_i)
# Dado un theta candidato, calculamos la probabilidad que el modelo asigna a
# que y = 1 en cada punto x_i de los datos.
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 1.2: probabilidades p_i para un theta de prueba ===")
theta_prueba = np.array([0.0, 1.0])
print(fc.compute_probs(theta_prueba, x))


# -----------------------------------------------------------------------------
# Ejercicio 1.3 — Verosimilitud L(theta)
# L(theta) = producto_i [ p_i^(y_i) * (1 - p_i)^(1 - y_i) ]
# Responde a la pregunta: si este theta fuera el verdadero, ¿qué tan probable
# sería haber observado exactamente los datos que observamos? Un theta es mejor
# que otro si le asigna mayor probabilidad a los datos reales.
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 1.3: verosimilitud para un theta de prueba ===")
print(fc.likelihood(theta_prueba, x, y))


# -----------------------------------------------------------------------------
# Ejercicio 1.4 — Evaluar la verosimilitud en toda la lista de candidatos
# Esta es la búsqueda por fuerza bruta: probamos los 5 thetas propuestos por el
# enunciado y anotamos qué verosimilitud alcanza cada uno.
# -----------------------------------------------------------------------------
theta_list = [
    np.array([-2.0, 0.5]),
    np.array([-1.0, 1.0]),
    np.array([0.0, 1.0]),
    np.array([0.0, 2.0]),
    np.array([1.0, 1.0])
]

print("\n=== Ejercicio 1.4: verosimilitud de cada candidato ===")
verosimilitudes = []

for theta in theta_list:
    L = fc.likelihood(theta, x, y)
    verosimilitudes.append(L)
    print(f"theta = {theta} -> L(theta) = {L:.6e}")


# -----------------------------------------------------------------------------
# Ejercicio 1.5 — Identificar el candidato que maximiza la verosimilitud
# np.argmax devuelve el índice del valor máximo dentro de la lista, y con ese
# índice recuperamos el theta correspondiente.
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 1.5: mejor candidato de la lista ===")
indice_optimo = np.argmax(verosimilitudes)

theta_optimo = theta_list[indice_optimo]
L_optimo = verosimilitudes[indice_optimo]

print(f"El theta que maximiza la verosimilitud es: {theta_optimo}")
print(f"Con L(theta) = {L_optimo:.6e}")


# -----------------------------------------------------------------------------
# Ejercicio 1.6 — Graficar los datos junto con la curva sigmoide ajustada
# Usamos un rango denso de x (x_curva) para que la sigmoide se dibuje suave, en
# lugar de solo en los 9 puntos observados.
# -----------------------------------------------------------------------------
x_curva = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)
y_curva = fc.compute_probs(theta_optimo, x_curva)

plt.figure(figsize=(8, 5))

# a) los datos originales (x_i, y_i)
plt.scatter(x, y, color='black', label='Datos $(x_i, y_i)$', zorder=3)

# b) la curva sigmoide con el theta óptimo de la lista finita
plt.plot(x_curva, y_curva, color='blue',
         label=fr'$\sigma(\hat\theta_0 + \hat\theta_1 x)$, $\hat\theta$={theta_optimo}')

plt.xlabel('x')
plt.ylabel('Probabilidad / y')
plt.title('Ejercicio 1.6 — Modelo logístico ajustado (búsqueda en lista finita)')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Umbral 0.5')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# -----------------------------------------------------------------------------
# Ejercicio 1.7 — Función escalón ("softmax") sobre el mismo plano
# Contrasta el clasificador probabilístico (sigmoide, decisión suave) contra el
# clasificador determinístico (escalón, decisión dura 0 ó 1). Ambos cambian de
# régimen en el mismo punto: donde theta0 + theta1 * x = 0.
# -----------------------------------------------------------------------------
theta0_hat, theta1_hat = theta_optimo[0], theta_optimo[1]

z_curva = theta0_hat + theta1_hat * x_curva
y_softmax = fc.softmax(z_curva)

plt.figure(figsize=(8, 5))

# a) los datos originales
plt.scatter(x, y, color='black', label='Datos $(x_i, y_i)$', zorder=3)

# b) la curva sigmoide
plt.plot(x_curva, y_curva, color='blue',
         label=fr'$\sigma(\hat\theta_0 + \hat\theta_1 x)$')

# c) la función escalón
plt.plot(x_curva, y_softmax, color='red', linestyle='--',
         label=fr'$\mathrm{{softmax}}(\hat\theta_0 + \hat\theta_1 x)$')

plt.xlabel('x')
plt.ylabel('Probabilidad / y')
plt.title('Ejercicio 1.7 — Sigmoide vs. función escalón')
plt.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =============================================================================
# EJERCICIO 2 — Minimización numérica del riesgo empírico
#
# Los puntos 2.1, 2.2 y 2.3 (sigmoide, pérdida logística y riesgo empírico) ya
# están implementados en src/functions.py como fc.sigmoid, fc.logistic_loss y
# fc.empirical_risk. Aquí los ponemos a trabajar.
# =============================================================================

# -----------------------------------------------------------------------------
# Ejercicio 2.4 — Minimizar el riesgo empírico con scipy.optimize.minimize
#
# ¡AQUÍ ESTABA EL ERROR "x is not defined"!
# La función objetivo debe definirse EN ESTE ARCHIVO, no en functions.py.
# Razón: scipy solo le pasa el vector theta, así que la función necesita ver a
# x e y desde algún lado. Si la definimos aquí, la función "captura" las
# variables x e y de este módulo (esto se llama clausura o closure). Si en
# cambio vive en functions.py, ahí adentro x e y simplemente no existen y
# Python lanza NameError.
# -----------------------------------------------------------------------------

# a) función objetivo: recibe solo theta y devuelve el riesgo empírico
def objetivo(theta):
    """
    Función objetivo para el optimizador. Recibe únicamente el vector
    theta = [theta0, theta1] y devuelve el riesgo empírico sobre los datos
    (x, y) definidos en este módulo.
    """
    return fc.empirical_risk(theta, x, y)


# b) valor inicial desde el cual arranca la búsqueda
theta_init = np.array([0.0, 0.0])

# c) optimización numérica
resultado = minimize(objetivo, theta_init)

# d) extraer el theta encontrado por el algoritmo
theta_hat = resultado.x

print("\n=== Ejercicio 2.4: resultado completo del optimizador ===")
print(resultado)


# -----------------------------------------------------------------------------
# Ejercicio 2.5 — Reportar el theta encontrado
# Compáralo mentalmente con el theta_optimo del Ejercicio 1: deberían ir en la
# misma dirección, pero este es el óptimo real sobre todo R^2, no solo sobre
# los 5 candidatos de la lista.
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 2.5: theta estimado ===")
print(f"theta_hat encontrado por el optimizador: {theta_hat}")
print(f"theta0_hat = {theta_hat[0]:.4f}")
print(f"theta1_hat = {theta_hat[1]:.4f}")
print(f"Riesgo empírico mínimo: {resultado.fun:.6f}")


# -----------------------------------------------------------------------------
# Ejercicio 2.6 — Graficar los datos con la curva sigmoide optimizada
# Nota: recalculamos y_curva con el NUEVO theta_hat, sobreescribiendo el valor
# que traía del Ejercicio 1. De aquí en adelante, y_curva es la curva del
# modelo optimizado numéricamente.
# -----------------------------------------------------------------------------
x_curva = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)
y_curva = fc.sigmoid(theta_hat[0] + theta_hat[1] * x_curva)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Datos $(x_i, y_i)$', zorder=3)
plt.plot(x_curva, y_curva, color='blue',
         label=fr'$\sigma(\hat\theta_0 + \hat\theta_1 x)$, $\hat\theta$=({theta_hat[0]:.2f}, {theta_hat[1]:.2f})')
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Umbral 0.5')
plt.xlabel('x')
plt.ylabel('Probabilidad / y')
plt.title('Ejercicio 2.6 — Modelo ajustado por minimización del riesgo empírico')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# -----------------------------------------------------------------------------
# Ejercicio 2.7 — Función escalón H(t) sobre el modelo optimizado
# Mismo contraste del Ejercicio 1.7, pero ahora con el theta óptimo real.
# -----------------------------------------------------------------------------
z_curva = theta_hat[0] + theta_hat[1] * x_curva
y_softmax = fc.softmax(z_curva)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Datos $(x_i, y_i)$', zorder=3)
plt.plot(x_curva, y_curva, color='blue', label=r'$\sigma(\hat\theta_0 + \hat\theta_1 x)$')
plt.plot(x_curva, y_softmax, color='red', linestyle='--', label=r'$H(\hat\theta_0 + \hat\theta_1 x)$')
plt.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
plt.xlabel('x')
plt.ylabel('Probabilidad / y')
plt.title('Ejercicio 2.7 — Sigmoide optimizada vs. función escalón')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =============================================================================
# EJERCICIO 3 — El modelo logístico como regresión lineal en el espacio logit
#
# El punto 3.1 es una demostración algebraica (va en el informe, no en código):
# si p(x) = sigma(theta0 + theta1*x), entonces logit(p(x)) = theta0 + theta1*x.
# Los puntos siguientes verifican esa identidad numéricamente.
# =============================================================================

# -----------------------------------------------------------------------------
# Ejercicio 3.2 — Calcular p_i con el theta_hat del Ejercicio 2
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 3.2: probabilidades ajustadas ===")
p = fc.sigmoid(theta_hat[0] + theta_hat[1] * x)
print("p_i =", p)


# -----------------------------------------------------------------------------
# Ejercicio 3.3 — Aplicar la transformación logit: z_i = logit(p_i)
# Por la identidad del punto 3.1, estos z_i deberían ser exactamente
# theta0_hat + theta1_hat * x_i, es decir, perfectamente lineales en x.
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 3.3: logits ===")
z = fc.logit(p)
print("z_i =", z)


# -----------------------------------------------------------------------------
# Ejercicio 3.4 — Graficar los puntos (x_i, z_i)
# Si la teoría es correcta, los puntos caen sobre una línea recta.
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(x, z, color='purple', label='Puntos $(x_i, z_i)$')
plt.xlabel('x')
plt.ylabel('z = logit(p)')
plt.title('Ejercicio 3.4 — Transformación logit de las probabilidades ajustadas')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# -----------------------------------------------------------------------------
# Ejercicio 3.5 — Ajustar una regresión lineal a (x_i, z_i)
# np.polyfit con deg=1 hace mínimos cuadrados y devuelve [pendiente, intercepto].
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 3.5: regresión lineal sobre los logits ===")
coef = np.polyfit(x, z, deg=1)
b_reg, a_reg = coef[0], coef[1]

print(f"Regresión lineal ajustada: z = {a_reg:.6f} + {b_reg:.6f} * x")
print(f"Intercepto (a) = {a_reg:.6f}")
print(f"Pendiente (b)  = {b_reg:.6f}")


# -----------------------------------------------------------------------------
# Ejercicio 3.6 — Comparar los coeficientes de la regresión con theta_hat
# Confirmación numérica de la identidad demostrada algebraicamente en 3.1:
# a debería coincidir con theta0_hat y b con theta1_hat.
# -----------------------------------------------------------------------------
print("\n=== Ejercicio 3.6: comparación de coeficientes ===")
print(f"  theta0_hat (Ejercicio 2) = {theta_hat[0]:.6f}   vs.   a (regresión) = {a_reg:.6f}")
print(f"  theta1_hat (Ejercicio 2) = {theta_hat[1]:.6f}   vs.   b (regresión) = {b_reg:.6f}")


# -----------------------------------------------------------------------------
# Ejercicio 3.7 — Reconstruir p(x) = sigma(a + b*x) y compararla con el modelo
# del Ejercicio 2. Las dos curvas deberían superponerse casi exactamente, ya
# que son algebraicamente equivalentes y parten de los mismos datos.
# -----------------------------------------------------------------------------
p_regresion = fc.sigmoid(a_reg + b_reg * x_curva)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='black', label='Datos $(x_i, y_i)$', zorder=3)
plt.plot(x_curva, y_curva, color='blue', linewidth=2.5,
         label=fr'$\sigma(\hat\theta_0+\hat\theta_1x)$ (Ejercicio 2)')
plt.plot(x_curva, p_regresion, color='green', linestyle='--',
         label=fr'$\sigma(a+bx)$ (vía regresión sobre logit)')
plt.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
plt.xlabel('x')
plt.ylabel('Probabilidad / y')
plt.title('Ejercicio 3.7 — Modelo logístico directo vs. reconstruido desde logit')
plt.legend()
plt.grid(alpha=0.3)
plt.show()




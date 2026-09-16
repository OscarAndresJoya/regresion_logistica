import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def sigmoid(t):
    """
    Calcula la función logística (sigmoide):
        sigma(t) = 1 / (1 + e^(-t))
    
    Parámetros
    ----------
    t : float o np.ndarray
        Valor (o arreglo de valores) de entrada.
    
    Retorna
    -------
    float o np.ndarray
        El valor de sigma(t), en el rango (0, 1).
    """
    return 1 / (1 + np.exp(-t))

def compute_probs(theta, x):
    """
    Dado un parámetro theta = (theta0, theta1) y un arreglo x,
    calcula las probabilidades p_i = sigma(theta0 + theta1 * x_i)
    para cada x_i en x.
    
    Parámetros
    ----------
    theta : np.ndarray de forma (2,)
        theta[0] = theta0 (intercepto)
        theta[1] = theta1 (pendiente)
    x : np.ndarray
        Arreglo de valores de entrada.
    
    Retorna
    -------
    np.ndarray
        Arreglo con las probabilidades p_i, misma forma que x.
    """
    theta0, theta1 = theta[0], theta[1]
    z = theta0 + theta1 * x
    return sigmoid(z)

def likelihood(theta, x, y):
    """
    Calcula la verosimilitud del modelo logístico:
        L(theta) = producto_i [ p_i^y_i * (1 - p_i)^(1 - y_i) ]
    
    Parámetros
    ----------
    theta : np.ndarray de forma (2,)
        Parámetros (theta0, theta1) del modelo.
    x : np.ndarray
        Arreglo de valores de entrada.
    y : np.ndarray
        Arreglo de etiquetas observadas (0 o 1), misma forma que x.
    
    Retorna
    -------
    float
        El valor de la verosimilitud L(theta).
    """
    p = compute_probs(theta, x)  # probabilidades p_i para cada x_i
    
    # Para cada observación, si y_i = 1 el factor es p_i;
    # si y_i = 0 el factor es (1 - p_i).
    factores = (p ** y) * ((1 - p) ** (1 - y))
    
    # La verosimilitud es el producto de todos los factores
    L = np.prod(factores)
    return L

def softmax(t):
    """
    Función escalón (a pesar del nombre 'softmax' pedido en el enunciado,
    NO es la softmax multiclase habitual, sino un umbral duro):
        softmax(t) = 0 si t < 0
        softmax(t) = 1 si t >= 0
    
    Parámetros
    ----------
    t : float o np.ndarray
        Valor (o arreglo de valores) de entrada.
    
    Retorna
    -------
    float o np.ndarray (mismo tipo que la entrada)
        0 si t < 0, 1 si t >= 0.
    """
    return np.where(t >= 0, 1, 0)

def sigmoid(t):
    """
    Calcula la función logística sigma(t) = 1 / (1 + e^(-t))
    """
    return 1 / (1 + np.exp(-t))

def logistic_loss(y_hat, y):
    """
    Calcula la pérdida logística puntual:
        l_log(y_hat, y) = -[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
    
    Parámetros
    ----------
    y_hat : float o np.ndarray
        Probabilidad predicha (debe estar en (0,1)).
    y : float o np.ndarray
        Etiqueta observada (0 o 1).
    
    Retorna
    -------
    float o np.ndarray
        Valor(es) de la pérdida logística.
    """
    # eps evita log(0) por errores numéricos si y_hat es muy cercano a 0 o 1
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def empirical_risk(theta, x, y):
    """
    Calcula el riesgo empírico del modelo logístico con parámetro theta,
    sobre la muestra (x, y).
    
    Parámetros
    ----------
    theta : np.ndarray de forma (2,)
        theta[0] = theta0, theta[1] = theta1
    x, y : np.ndarray
        Datos de entrada y etiquetas.
    
    Retorna
    -------
    float
        El riesgo empírico R_S_hat(theta).
    """
    theta0, theta1 = theta[0], theta[1]
    y_hat = sigmoid(theta0 + theta1 * x)   # probabilidades predichas
    perdidas = logistic_loss(y_hat, y)      # pérdida por cada observación
    return np.mean(perdidas) 

def objetivo(theta):
    return empirical_risk(theta, x, y)

def softmax(t):
    """
    Función escalón:
        H(t) = 0 si t < 0
        H(t) = 1 si t >= 0
    (Se nombra 'softmax' por instrucción del enunciado, aunque
    no corresponde a la softmax multiclase habitual)
    """
    return np.where(t >= 0, 1, 0)

def logit(p):
    """
    Calcula la función logit:
        logit(p) = log(p / (1 - p))
    
    Parámetros
    ----------
    p : float o np.ndarray
        Probabilidad(es) en (0, 1).
    
    Retorna
    -------
    float o np.ndarray
        El valor de logit(p).
    """
    return np.log(p / (1 - p))
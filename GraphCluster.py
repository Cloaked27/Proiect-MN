import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
def generate_graph():
    G = nx.stochastic_block_model(
        sizes=[50, 50, 50],
        p=[[0.7, 0.05, 0.05],
           [0.05, 0.7, 0.05],
           [0.05, 0.05, 0.7]],
        seed = 42
    )
    return G

G = generate_graph()
A = nx.to_numpy_array(G)
D = np.diag(np.sum(A, axis=1))
L = D - A


def HD(x): #householder, functie pe care am facut-o pentru a introduce 0 sub prima pozitie din vector
    x = x.astype(np.float64)
    sigma = np.linalg.norm(x) 
    if sigma == 0:
        return x, 0.0
    if x[0] >= 0:
        alpha = -sigma
    else:
        alpha = sigma
    v = x.copy()
    v[0] = v[0] - alpha
    beta = 2.0 / (v @ v)
    return v, beta  #O(n)
 
def Hrd(Matrix, u, beta):   #inmultim la dreapta cu matricea householder
    y = Matrix @ u
    # M = M - beta * y * u^T
    Matrix -= beta * np.outer(y, u)
    return Matrix  #O(n^2)

def Givens(a): #rotatii Givens pe care le vom folosi tot pentru a introduce 0, dar in pozitiile de sub diagonala principala
    r = np.hypot(a[0], a[1])
    if( r < 1e-20):
        return a, 1, 0
    if abs(a[1]) > abs(a[0]):
        r = np.sign(a[1])*r
    else:
        r = np.sign(a[0])*r
    c = a[0]/r
    s = a[1]/r
    b = a.copy()
    b[0] = r
    b[1] = 0
    return b,c,s #O(1)


def tridiagonalizare_finala(A):
    A = A.astype(np.float64).copy()
    n = A.shape[0]
    Q = np.eye(n)    
    for k in range(n - 2):
        x = A[k+1:, k]
        if np.linalg.norm(x) < 1e-10:
            continue
        u, beta = HD(x)
        if beta == 0:
            continue
        # Actualizare A[k+1:, k+1:]
        A_sub = A[k+1:, k+1:]
        p = beta * (A_sub @ u)
        k_val = beta * np.dot(p, u) / 2.0
        w = p - k_val * u
        A[k+1:, k+1:] -= (np.outer(u, w) + np.outer(w, u))  
        # Actualizare 
        Q_sub = Q[:, k+1:]  # toate rândurile, coloanele k+1
        Q[:, k+1:] = Hrd(Q_sub, u, beta)  
        norm_val = np.linalg.norm(x)
        if x[0] >= 0: 
            norm_val = -norm_val
        A[k+1, k] = norm_val
        A[k, k+1] = norm_val
    f = np.diag(A)
    g = np.array([A[i+1, i] for i in range(n-1)])
    return f, g, Q  #O(n^3)

def IT_QRsim(f, g):
    f = f.copy().astype(float)
    g = g.copy().astype(float)
    n = len(f)
    h = np.zeros(n)
    c = np.zeros(n-1)
    s = np.zeros(n-1)
    alpha = (f[n-2]-f[n-1])/2
    beta = g[n-2] * g[n-2]
    sign = 1 if alpha >= 0 else -1
    numitor = alpha + sign*np.sqrt(alpha*alpha+beta)
    if abs(numitor) < 1e-15:
        miu = f[n-1]
    else:
        miu = f[n-1] -(beta/numitor)
    f = f - miu  #echivalentul lui T = T- miu * I
    h[0] = g[0]
    for i in range (n-1):
        v = np.array([f[i], g[i]])
        [w, c[i], s[i]] = Givens(v) #aici construim Givens-ul care sa ne anuleze g[i] 
        f[i] = w[0]
        tau = h[i]
        h[i] = c[i]*h[i] + s[i]*f[i+1]
        f[i+1] = -s[i] * tau + c[i]*f[i+1] #rezultatele oricarei rotatii Givens
        if i < n-2:
            h[i+1] = c[i]*g[i+1]
    for i in range(n-1): #aplicam rotatiile din dreapta, RQ
        f[i] = c[i]*f[i] + s[i]*h[i]
        copie_urm = f[i+1]
        g[i] = s[i]*f[i+1]
        f[i+1] = c[i]*f[i+1]
    f = f+ miu
    return f, g #O(n)

def QR_iterativ(f, g):
    for _ in range(5000):
        if np.max(np.abs(g)) < 1e-9:
            break
        f, g =IT_QRsim(f,g)
    return f

def VectorPropriu(A, x):
    n = A.shape[0]
    I = np.eye(n)
    np.random.seed()
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    B = A - x* I
    for _ in range(10):
        v = np.linalg.solve(B, v)
        v = v / np.linalg.norm(v)
    return v

f, g, Q = tridiagonalizare_finala(L)

valoriproprii = QR_iterativ(f, g)
valoriproprii = np.sort(valoriproprii)

lambda2 = valoriproprii[1]
lambda3 = valoriproprii[2]

v2 = VectorPropriu(L , lambda2)
v3 = VectorPropriu(L , lambda3)

def clustering_3_semne_simplu(v2, v3):
    clusters = np.zeros(len(v2), dtype=int)

    for i in range(len(v2)):
        if v2[i] >= 0 and v3[i] <= 0:
            clusters[i] = 0
        elif v2[i] < 0 and v3[i] >= 0:
            clusters[i] = 1
        else:
            clusters[i] = 2

    return clusters
labels_3 = clustering_3_semne_simplu(v2, v3)


def clustering_2(v2):
    clusters = np.zeros(len(v2), dtype=int)
    for i in range(len(v2)):
        if v2[i] >= 0:
            clusters[i] = 1
        else:
            clusters[i] = 0

    return clusters

labels_2 = clustering_2(v2)

#print("Toate valorile proprii (primele 10):")
#print(valoriproprii[:10])
T = np.diag(f) + np.diag(g, -1) + np.diag(g, 1)

#print("Verificare tridiagonalizare:")
#print("||L - Q*T*Q^T||:", np.linalg.norm(L - Q @ T @ Q.T))
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color='lightgray', node_size=60, edge_color='gray', with_labels=False)
plt.title("Graful Original")
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(v2, np.zeros_like(v2), c=labels_2, cmap='coolwarm', s=30, edgecolors='black')
plt.title("Separare pe axa v2")
plt.yticks([]) 
plt.xlabel("v2")
plt.grid(True, axis='x', alpha=0.3)

# Proiectia 2D (v2 vs v3)
plt.subplot(1, 2, 2)
plt.scatter(v2, v3, c=labels_3, cmap='viridis', s=30, edgecolors='black')
plt.title("Separare pe v2 și v3")
plt.xlabel("v2")
plt.ylabel("v3")
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

sc2 = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans', random_state=42)
sklearn_labels_2 = sc2.fit_predict(nx.to_numpy_array(G))

sc3 = SpectralClustering(n_clusters=3, affinity='precomputed', assign_labels='kmeans', random_state=42)
sklearn_labels_3 = sc3.fit_predict(nx.to_numpy_array(G))

ari_2 = adjusted_rand_score(labels_2, sklearn_labels_2)
ari_3 = adjusted_rand_score(labels_3, sklearn_labels_3)
print(f"Scor ARI (2 Clustere): {ari_2:.4f}")
print(f"Scor ARI (3 Clustere): {ari_3:.4f}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
nx.draw(G, pos, node_color=labels_2, cmap='coolwarm', node_size=40, edge_color='gray')
plt.title(f"2 Clustere: Codul meu (ARI: {ari_2:.2f})")

plt.subplot(2, 2, 2)
nx.draw(G, pos, node_color=sklearn_labels_2, cmap='coolwarm', node_size=40, edge_color='gray')
plt.title("2 Clustere: functia Python")

plt.subplot(2, 2, 3)
nx.draw(G, pos, node_color=labels_3, cmap='viridis', node_size=40, edge_color='gray')
plt.title(f"3 Clustere: Codul meu (ARI: {ari_3:.2f})")

plt.subplot(2, 2, 4)
nx.draw(G, pos, node_color=sklearn_labels_3, cmap='viridis', node_size=40, edge_color='gray')
plt.title("3 Clustere: functia Python")

plt.tight_layout()
plt.show()  
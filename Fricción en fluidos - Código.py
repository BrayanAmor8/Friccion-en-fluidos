# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:24:18 2021

@author: Brayan
"""

#Importanding bibliotecas

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# Importando datos de tracker
datos_sc = pd.read_excel(r"C:\Users\braya\Documents\Brayan\Introducción a la física\Código - Fricción en fluidos\Servilleta Comprimida, Datos (t,y,v,a).xlsx", usecols=("F:H"), skiprows = 3, nrows = 19)
datos_se = pd.read_excel(r"C:\Users\braya\Documents\Brayan\Introducción a la física\Código - Fricción en fluidos\Aceleraciones servilleta extendida.xlsx", usecols=("A:BB"), skiprows = 8)
 
#Definición de variables para la servilleta arrugada

Vi = 0
g  = 9.78
h  = 2
Tmax = math.sqrt((2*h)/g) #Tiempo máximo de caída


#Funciones para calcular la velocidad y la posición en función del tiempo

def Vs(t):
    V1 = g*t
    return V1

def Y(t):
    Y1 = h-(g * t**2)/2
    return Y1

time = np.arange(0,Tmax+0.01,0.01) # Tiempos

cv = Vs(time)
cy = Y(time)

#fuentes bonitas para las gráficas

font = {"family" : "serif",
        "color"  : "darkred",
        "weight" : "normal",
        "size"   : 16,
        }

font2 = {"family" : "serif",
         "color"  : "black",
         "weight" : "bold",
         "size"   : 12,
         }

#Gráficas teóricas de caída libre
#Velocidad vs tiempo
plt.figure()
plt.plot(time,cv, "k-", linewidth = 2, markersize = 10)
plt.ylabel("Velocidad (m/s)", fontdict = font)
plt.xlabel("Tiempo (s)", fontdict = font)
plt.title("Servilleta Arrugada", fontdict = font)
plt.text(-0.02, 5.7,"$V_{f} = 6.25 m/s$", fontdict = font2)
plt.text(0.495, 0, "$T_{max} = 0.639s$", fontdict = font2)
plt.grid()
plt.show()

#Posición vs tiempo
plt.figure()
plt.plot(time,cy, "k-", linewidth= 2, markersize=10)
plt.ylabel("Altura (m)", fontdict = font)
plt.xlabel("Tiempo (s)", fontdict = font)
plt.title("Servilleta Arrugada", fontdict = font)
plt.grid()
plt.show()

#Datos obtenidos del experimento: servilleta arrugada

datos_t_sc = np.array(datos_sc["t"])
datos_y_sc = np.array(datos_sc["y"])
datos_v_sc = np.array(datos_sc["v"])

#Gráfica experimento - servilleta cerrada
plt.figure()
plt.plot(datos_t_sc,datos_v_sc,"c-", linewidth = 2, markersize = 10)
plt.grid()
plt.ylabel("Velocidad (m/s)", fontdict = font)
plt.xlabel("Tiempo (s)", fontdict = font)
plt.title("Servilleta Arrugada - Experimento", fontdict = font)
plt.grid()
plt.show()

plt.figure()
plt.plot(datos_t_sc, datos_y_sc, "c-", linewidth = 2, markersize = 10)
plt.grid()
plt.ylabel("Altura (m)", fontdict = font)
plt.xlabel("Tiempo (s)", fontdict = font)
plt.title("Servilleta Arrugada", fontdict = font)
plt.grid()
plt.show()

#Comparación de Datos

#Altura vs tiempo - Servilleta Arrugada
plt.figure()
plt.plot(datos_t_sc,datos_y_sc, "c-", linewidth = 2, markersize=10, label= "Experimento")
plt.plot(time, cy, "k-", linewidth = 2, markersize=10, label = "Simulación")
plt.legend(loc="upper right")
plt.grid()
plt.ylabel("Altura (m)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Arrugada - Comparación", fontdict= font)
plt.show()

#Velocidad vs Tiempo - Servilleta Arrugada

plt.figure()
plt.plot(datos_t_sc,datos_v_sc, "c-", linewidth = 2, markersize=10, label= "Experimento")
plt.plot(time, cv, "k-", linewidth = 2, markersize=10, label = "Simulación")
plt.legend(loc="upper left")
plt.grid()
plt.ylabel("Velocidad (v)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Arrugada - Comparación", fontdict= font)
plt.show()

#Cálculo de la varianza

cv2 =  Vs(datos_t_sc)
cy2 =  Y(datos_t_sc)
varianza_sc = 0
varianza_sc_y = 0

for i in range(0,len(datos_v_sc+1)):
    var = (datos_v_sc[i]  - cv2[i])**2
    varianza_sc =  varianza_sc + var    
varianza_sc = varianza_sc/len(cv2)
print(varianza_sc)

for i in range(0,len(datos_y_sc+1)):
    var = (datos_y_sc[i]  - cy2[i])**2
    varianza_sc_y =  varianza_sc_y + var    
varianza_sc_y = varianza_sc_y/len(cv2)
print(varianza_sc_y)

#Datos del experimento necesarios para calcular K

tt = np.array(datos_se["t"])
yy = np.array(abs(datos_se["y"])-2)
vv = np.array(datos_se["v"])
aa = np.array(datos_se["a"])

datos_se_t = tt[:-2]
datos_se_y = yy[:-2]
datos_se_v = vv[:-2]
datos_se_a = aa[:-2]

#Definición de variables: Servilleta Extendida
time2= 3
taza_cambio = time2/len(datos_se_t)
T = np.arange(0, time2, taza_cambio)
V = np.zeros(len(T))
X = np.zeros(len(T))
a = np.zeros(len(T))
ac = np.zeros(len(datos_se_t))
X[-1] = 2
a[0] ,ac[0], ac[2] = g, g, g #Usada para el bucle donde calculo k
m = 0.0007

#Estimación de k con los datos de tracker
K = np.zeros(len(datos_se_t)-1)
for i in range(2,len(datos_se_t)-1): #Empieza en 2 pq en el primer instante es indeterminado
     ac[i] = (datos_se_v[i]-datos_se_v[i-1])/(datos_se_t[i]-datos_se_t[i-1])
     K[i] = m*(g-ac[i])/(datos_se_v[i])
     

# =============================================================================
# 
   
# #Modelo teórico con K definida
# for j in range (2,len(datos_se_t)-1):
#     V[i] = V[i-1] + a[i-1]*(T[i]-T[i-1])
#     a[i] = g-((K[i]/m)*(V[i]))
#     X[i] = (((X[i-1] + V[i-1] * (T[i]-T[i-1]) + a[i-1]*((T[i]-T[i-1])**2)/2)))
# =============================================================================


#Pruebas con K constante 
k = sum(K)/len(K)
print("k es igual a ", k)
for b in range (1,len(datos_se_t)): 
    a[b] = g - ((k/m)*V[b-1])
    V[b] = V[b-1] + a[b-1]*(T[b]-T[b-1])
    X[b] = (((X[b-1] + V[b-1] * (T[b] - T[b-1]) + a[b-1]*((T[b]-T[b-1])**2)/2)))

      

#Graficas teóricas
plt.figure()
plt.plot(T,V, "m-", linewidth = 2, markersize = 10)
plt.ylabel("Velocidad (m/s)", fontdict= font)
plt.xlabel("Tiempo(s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()

plt.figure()
plt.plot(T, 2-X, "m-", linewidth = 2, markersize=10)
plt.ylabel("Distancia recorrida (m)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()

#Datos experimento servilleta extendida
tt = np.array(datos_se["t1"]); datos_se_t_1 = tt[:-16]
yy = np.array(abs(2-datos_se["y1"])); datos_se_y_1 = yy[:-16]
vv = np.array(datos_se["v1"]); datos_se_v_1 = vv[:-16]

tt = np.array(datos_se["t2"]); datos_se_t_2 = tt[:-1]
yy = np.array(abs(2-datos_se["y2"])); datos_se_y_2 = yy[:-1]
vv = np.array(datos_se["v2"]); datos_se_v_2 = vv[:-1]

tt = np.array(datos_se["t3"]); datos_se_t_3 = tt[:-16]
yy = np.array(abs(2-datos_se["y3"])); datos_se_y_3 = yy[:-16]
vv = np.array(datos_se["v3"]); datos_se_v_3 = vv[:-16]

tt = np.array(datos_se["t4"]); datos_se_t_4 = tt
yy = np.array(abs(2-datos_se["y4"])); datos_se_y_4 = yy
vv = np.array(datos_se["v4"]); datos_se_v_4 = vv

tt = np.array(datos_se["t5"]); datos_se_t_5 = tt[:-16]
yy = np.array(abs(2-datos_se["y5"])); datos_se_y_5 = yy[:-16]
vv = np.array(datos_se["v5"]); datos_se_v_5 = vv[:-16]

tt = np.array(datos_se["t6"]); datos_se_t_6 = tt[:-19]
yy = np.array(abs(2-datos_se["y6"])); datos_se_y_6 = yy[:-19]
vv = np.array(datos_se["v6"]); datos_se_v_6 = vv[:-19]

tt = np.array(datos_se["t7"]); datos_se_t_7 = tt[:-3]
yy = np.array(abs(2-datos_se["y7"])); datos_se_y_7 = yy[:-3]
vv = np.array(datos_se["v7"]); datos_se_v_7 = vv[:-3]

tt = np.array(datos_se["t8"]); datos_se_t_8 = tt[:-21]
yy = np.array(abs(2-datos_se["y8"])); datos_se_y_8 = yy[:-21]
vv = np.array(datos_se["v8"]); datos_se_v_8 = vv[:-21]

tt = np.array(datos_se["t9"]); datos_se_t_9 = tt[:-10]
yy = np.array(abs(2-datos_se["y9"])); datos_se_y_9 = yy[:-10]
vv = np.array(datos_se["v9"]); datos_se_v_9 = vv[:-10]

tt = np.array(datos_se["t10"]); datos_se_t_10 = tt[:-3]
yy = np.array(abs(2-datos_se["y10"])); datos_se_y_10 = yy[:-3]
vv = np.array(datos_se["v10"]); datos_se_v_10 = vv[:-3]

#Comparación de Datos
plt.figure()
plt.plot(datos_se_t_1,datos_se_y_1, "c-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_2,datos_se_y_2, "b-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_3,datos_se_y_3, "g-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_4,datos_se_y_4, "m-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_5,datos_se_y_5, "y-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_6,datos_se_y_6, "b-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_7,datos_se_y_7, "w-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_8,datos_se_y_8, "r-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_9,datos_se_y_9, "c-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_10,datos_se_y_10, "m-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t, abs(datos_se_y), "k-", linewidth = 2, markersize=5, label = "Promedio")
plt.legend(loc="upper right")
plt.grid()
plt.ylabel("Altura (m)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Comparación de datos - Experimento", fontdict= font)
plt.show()


plt.figure()
plt.plot(datos_se_t_1,datos_se_v_1, "c-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_2,datos_se_v_2, "b-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_3,datos_se_v_3, "g-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_4,datos_se_v_4, "m-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_5,datos_se_v_5, "y-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_6,datos_se_v_6, "b-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_7,datos_se_v_7, "w-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_8,datos_se_v_8, "r-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_9,datos_se_v_9, "c-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t_10,datos_se_v_10, "m-", linewidth = 0.5, markersize=5)
plt.plot(datos_se_t, datos_se_v, "k-", linewidth = 2, markersize=5, label = "Promedio")
plt.legend(loc="upper right")
plt.grid()
plt.ylabel("Velocidad (m/s)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Comparación de datos - Experimento", fontdict= font)
plt.show()


#Distancia vs Tiempo - Servilleta Extendida

plt.figure()
plt.plot(datos_se_t, abs(datos_se_y), "-", linewidth = 2, markersize=10, label= "Experimento")
plt.plot( T, 2-X, "m-", linewidth = 2, markersize=10, label = "Simulación")
plt.legend(loc="upper right")
plt.grid()
plt.ylabel("Altura (m)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida - Comparación", fontdict= font)
plt.show()

#Velocidad vs Tiempo - Servilleta Extendida

plt.figure()
plt.plot(datos_se_t,datos_se_v, "c-", linewidth = 2, markersize=10, label= "Experimento")
plt.plot(T, V, "k-", linewidth = 2, markersize=10, label = "Simulación")
plt.legend(loc="upper right")
plt.grid()
plt.ylabel("Velocidad (v)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida - Comparación", fontdict= font)
plt.show()


#Graficando K
K1 = K.tolist()
t1 = datos_se_t.tolist() 
t1.pop(-1)
plt.bar(t1,K1, width=0.02)
plt.ylabel("Resistencia del aire (k)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida - Comparación", fontdict= font)
plt.show()


#Varianza de la servilleta extendida

varianza_se_v = 0
varianza_se_X = 0

for i in range(0, len(datos_se_v+1)):
    var = (datos_se_v[i]-V[i])**2
    varianza_se_v = varianza_se_v + var
varianza_se_v = varianza_se_v/len(V)
print(varianza_se_v)

for i in range(0, len(datos_se_y+1)):
    var = (datos_se_y[i]-X[i])**2
    varianza_se_X = varianza_se_X + var
varianza_se_X = varianza_se_X/len(V)
print(varianza_se_X)

































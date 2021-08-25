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

#Datos servilleta arrugada del documento "Servilleta Comprimida, Datos (t,y,v,a).xlsx"
datos_sc = pd.read_excel(r"C:\Users\braya\Documents\Brayan\Introducción a la física\Código - Fricción en fluidos\Servilleta Comprimida, Datos (t,y,v,a).xlsx", usecols=("F:H"), skiprows = 3, nrows = 19)
#Datos servilleta extendida del documento "Aceleraciones servilleta extendida.xlsx"
datos_se = pd.read_excel(r"C:\Users\braya\Documents\Brayan\Introducción a la física\Código - Fricción en fluidos\Aceleraciones servilleta extendida.xlsx", usecols=("A:BB"), skiprows = 8)
 
#Definición de variables para la servilleta arrugada

Vi = 0
g  = 9.78
h  = 2
m = 0.0007
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
print("Varianza en V para la servilleta cerrada:", varianza_sc)

for i in range(0,len(datos_y_sc+1)):
    var = (datos_y_sc[i]  - cy2[i])**2
    varianza_sc_y =  varianza_sc_y + var    
varianza_sc_y = varianza_sc_y/len(cv2)
print("Varianza en X para la servilleta cerrada:", varianza_sc_y)



#Datos del experimento necesarios para calcular K

Tiempo, Altura, Velocidad = [], [], []
for i in range (0,11):
    n = str(i+1)
    Tiempo.append(np.array(datos_se["t{}".format(n)]))
    Tiempo[i] = [x for x in Tiempo[i] if np.isnan(x) == False]
    Altura.append(np.array(abs(2-datos_se["y{}".format(n)])))
    Altura[i] = [x for x in Altura[i] if np.isnan(x) == False]
    Velocidad.append(np.array(datos_se["v{}".format(n)]))
    Velocidad[i] = [x for x in Velocidad[i] if np.isnan(x) == False]
    
#Hallando k


K, T, V, X, a = [], [], [], [], []
Resist = []

for j in range(0,10):
    K.append(np.zeros(len(Tiempo[j])))
    time2= Tiempo[j][-1]
    taza_cambio = time2/len(Tiempo[j])
    T.append(np.arange(0, time2, taza_cambio))
    V.append(np.zeros(len(T[j])))
    X.append(np.zeros(len(T[j])))
    a.append([g for i in range(len(T[j]))])
    for i in range(2,len(Tiempo[j])):
        Delta_vel = Velocidad[j][i]-Velocidad[j][i-1]
        Delta_tiempo = Tiempo[j][i]-Tiempo[j][i-1]
        a[j][i] = (Delta_vel)/(Delta_tiempo)
        K[j][i] = m*(g-a[j][i])/Velocidad[j][i]
    k = sum(K[j]/len(K[j]))
    print("Modelo {} para una k de:".format(j+1), k)
    
    Resist.append(k)
    for l in range(1,len(T[j])):
        a[j][l] = g - ((k)/m)*V[j][l-1]
        V[j][l] = V[j][l-1] + (a[j][l]*(T[j][l]-T[j][l-1]))
        X[j][l] = (X[j][l-1] + V[j][l] * (T[j][l]-T[j][l-1]) + a[j][l-1]*((T[j][l] - T[j][l-1])**2)/2)
    
  
    

     
plt.figure()
plt.plot(T[0], a[0], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 1")
plt.plot(T[1], a[1], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 2")
plt.plot(T[2], a[2], "g.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 3")
plt.plot(T[3], a[3], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 4")
plt.plot(T[4], a[4], "y.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 5")
plt.plot(T[5], a[5], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 6")
plt.plot(T[6], a[6], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 7")
plt.plot(T[7], a[7], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 8")
plt.plot(T[8], a[8], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 9")
plt.plot(T[9], a[9], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 10")
plt.legend(loc = "best", bbox_to_anchor=(0.8, 0.5, 0.5, 0.5))
plt.ylabel("Aceleración ($m/s^{2}$)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()      

plt.figure()

# plt.plot(T[0], V[0], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 1")
# plt.plot(T[1], V[1], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 2")
# plt.plot(T[2], V[2], "g.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 3")
# plt.plot(T[3], V[3], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 4")
plt.plot(T[4], V[4], "y-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 5")
plt.plot(T[5], V[5], "b-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 6")
# plt.plot(T[6], V[6], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 7")
plt.plot(T[7], V[7], "c-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 8")
plt.plot(T[8], V[8], "r-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 9")
plt.plot(T[9], V[9], "m-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 10")
plt.legend(loc = "best")
plt.ylabel("Velocidad (m/s)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()


plt.figure()
plt.plot(T[0], 2-X[0], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 1")
plt.plot(T[1], 2-X[1], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 2")
plt.plot(T[2], 2-X[2], "g.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 3")
plt.plot(T[3], 2-X[3], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 4")
plt.plot(T[4], 2-X[4], "y.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 5")
plt.plot(T[5], 2-X[5], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 6")
plt.plot(T[6], 2-X[6], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 7")
plt.plot(T[7], 2-X[7], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 8")
plt.plot(T[8], 2-X[8], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 9")
plt.plot(T[9], 2-X[9], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 10")
plt.legend(loc = "best", bbox_to_anchor=(0.8, 0.5, 0.5, 0.5))
plt.ylabel("Altura (m/s)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()





plt.figure()
plt.plot(Tiempo[0], Velocidad[0], "c-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 1")
plt.plot(Tiempo[1], Velocidad[1], "b-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 2")
plt.plot(Tiempo[2], Velocidad[2], "g-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 3")
plt.plot(Tiempo[3], Velocidad[3], "m-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 4")
plt.plot(Tiempo[4], Velocidad[4], "y-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 5")
plt.plot(Tiempo[5], Velocidad[5], "b-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 6")
plt.plot(Tiempo[6], Velocidad[6], "r-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 7")
plt.plot(Tiempo[7], Velocidad[7], "c-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 8")
plt.plot(Tiempo[8], Velocidad[8], "r-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 9")
plt.plot(Tiempo[9], Velocidad[9], "m-", linewidth = 1, markersize=3, alpha=0.3, label = "Exp. 10")
plt.plot(T[4], V[4], "y-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 5")
plt.plot(T[5], V[5], "b-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 6")
# plt.plot(T[6], V[6], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Modelo 7")
plt.plot(T[7], V[7], "c-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 8")
plt.plot(T[8], V[8], "r-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 9")
plt.plot(T[9], V[9], "m-", linewidth = 2, markersize=3, alpha=1, label = "Modelo 10")
plt.plot(Tiempo[10], Velocidad[10], "k-", linewidth = 2, markersize = 3, label = "Promedio")
plt.legend(loc = "best",bbox_to_anchor=(0.8, 0.5, 0.5, 0.5))
plt.ylabel("Velocidad (m/s)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()

plt.figure()
plt.plot(Tiempo[0], Altura[0], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 1")
plt.plot(Tiempo[1], Altura[1], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 2")
plt.plot(Tiempo[2], Altura[2], "g.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 3")
plt.plot(Tiempo[3], Altura[3], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 4")
plt.plot(Tiempo[4], Altura[4], "y.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 5")
plt.plot(Tiempo[5], Altura[5], "b.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 6")
plt.plot(Tiempo[6], Altura[6], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 7")
plt.plot(Tiempo[7], Altura[7], "c.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 8")
plt.plot(Tiempo[8], Altura[8], "r.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 9")
plt.plot(Tiempo[9], Altura[9], "m.-", linewidth = 1, markersize=3, alpha=0.5, label = "Exp. 10")
plt.plot(Tiempo[10], Altura[10], "k-", linewidth = 2, markersize = 3, label = "Promedio")
plt.legend(loc = "best",bbox_to_anchor=(0.8, 0.5, 0.5, 0.5))
plt.ylabel("Altura (m/s)", fontdict= font)
plt.xlabel("Tiempo (s)", fontdict= font)
plt.title("Servilleta Extendida", fontdict= font)
plt.grid()
plt.show()




#Graficando K
Modelos = ["1","2","3","4","5","6","7","8","9","10"]
plt.bar(Modelos, Resist, width=0.8)
plt.ylabel("Resistencia del aire", fontdict= font)
plt.xlabel("Experimento", fontdict= font)
plt.title("Estimación de k", fontdict= font)
plt.show()


#Varianza de la servilleta extendida

for j in range(0,10):    
    var = []
    varianza_se_v = 0   
    varianza_se_X = 0

    for i in range(0, len(Velocidad[j])):
        var.append(Velocidad[j][i]-V[j][i])
        varianza= (Velocidad[j][i]-V[j][i])**2
        varianza_se_v = varianza_se_v + varianza
   
    
    varianza_se_v = varianza_se_v
    print("La varianza para el experimento {} en V es de".format(j+1), varianza_se_v)

    plt.scatter(T[j], var, alpha=1)
    plt.axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    plt.title("Análisis de residuos del modelo {}".format(j+1), fontweight = "bold", fontdict= font)
    plt.xlabel('Tiempo(s)', fontdict= font)
    plt.ylabel('Residuo', fontdict= font)   
    plt.show()



for j in range(0, 10):
    var = []
    for i in range(0, len(Altura[j])):
        
        var.append(Altura[j][i]-X[j][i])
        varianza = (Altura[j][i]-X[j][i])**2
        varianza_se_X = varianza_se_X + varianza
    # plt.scatter(T[j], var)
    # plt.show()
    varianza_se_X = varianza_se_X/len(V)
    print("La varianza para el experimento {} en X es de".format(j+1), varianza_se_X)

































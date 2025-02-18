#Cálculo de los efectos de la temperatura y la presión barométrica en la medición de CO2.
#ppm CO2 corregido = ppm CO2 medido ((Tmedidopref) / (pmedido*Tref))


#-----Datos extraidos-----

Tactual=35 #°C
#Temperatura actual en °C
ppmC02medido=900 #ppm
#Valor de ppm C02 Medido

#-----Variables-----
Pmedida=1000 #hPa
#pmedida = Presión actual, en las mismas unidades que la presión de referencia (no corregida al nivel del mar)

Tref=298.15 #°K
#Tref = temperatura de referencia, generalmente 25°C, convertida a absoluta (298,15 para °C)

Tmedida=Tactual+273.15 #°K
#Tmedida = temperatura absoluta actual, °C + 273,15

Pref=1013.207 #hPa
# Pref=29.92 #Hg / Pref=760 #mm Hg / Pref=14.6959 #psi
#pref = presión barométrica de referencia, normalmente a nivel del mar

#-----Cálculos-----

ppmC02corregido = ppmC02medido * ((Tmedida*Pref)/(Pmedida*Tref))

#-----Imprimir Valores-----

print("Valor calculado de PPM C02 corredido=",ppmC02corregido)


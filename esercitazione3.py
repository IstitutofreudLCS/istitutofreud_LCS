#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
animali = ['Elefanti', 'Leoni', 'Tigri', 'Scimmie', 'Zebre']
numero_animali = [4, 3, 2, 8, 15]
plt.bar(animali, numero_animali, color="lightblue")
plt.show()


# In[ ]:





# In[2]:


import matplotlib.pyplot as plt

animali = ['Elefanti', 'Leoni', 'Tigri', 'Scimmie', 'Zebre']
numero_animali = [4, 3, 2, 8, 5]

plt.bar(animali, numero_animali, color='skyblue')
plt.title('Numero di animali in uno zoo')
plt.xlabel('Animali')
plt.ylabel('Numero')
plt.show()


# In[3]:


mese = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
temperatura_media = [10, 12, 15, 18, 22]
plt.plot(mese, temperatura_media, marker='*', linestyle='-', color='lightpink')
plt.title('Andamento delle temperature medie mensili')
plt.xlabel('Mese')
plt.ylabel('Temperatura Media (°C)')
plt.grid(True)
plt.show()


# In[4]:


mese = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
temperatura_media = [10, 12, 15, 18, 22]
plt.plot(mese, temperatura_media, marker='o', linestyle='--', color='lightpink')
plt.title('Andamento delle temperature medie mensili')
plt.xlabel('Mese')
plt.ylabel('Temperatura Media (°C)')
plt.grid(True,axis="y")
plt.show()


# In[6]:


mese = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
temperatura_media = [10, 12, 15, 18, 22]
plt.plot(temperatura_media, mese, marker='o', linestyle='-', color='blue')
plt.title('Andamento delle temperature medie mensili')
plt.xlabel('Mese')
plt.ylabel('Temperatura Media (°C)')
plt.grid(True)
plt.show()



# In[8]:


vendite_mensili={
    "gen":1200,
    "feb":1000,
    "mar":3300,
    "apr":4555
}

plt.bar(vendite_mensili.keys(),vendite_mensili.values(),color="blue")
plt.show()


# In[7]:


colori = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
mese = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
temperatura_media = [10, 12, 15, 18, 22]
plt.pie(temperatura_media, labels=mese, colors=colori)
plt.title('Percentuale di temperatura media mensile')
plt.show()


# In[10]:


colori = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
temperatura_mesi={
    'Gennaio':10, 
    'Febbraio':12, 
    'Marzo':15,
    'Aprile':18,
    'Maggio':22
    
}
plt.pie(temperatura_mesi.values(), labels=temperatura_mesi.keys(), colors=colori)
plt.title('Percentuale di temperatura media mensile')
plt.show()


# In[ ]:





# In[11]:


età = [14, 15, 16, 17, 18, 19]
altezza = [160, 165, 170, 175, 180, 185]
plt.scatter(età, altezza, color='red', marker='o')
plt.title('Scatter Plot - Età vs Altezza')
plt.xlabel('Età')
plt.ylabel('Altezza (cm)')
plt.grid(True, axis='y')
plt.show()



# In[ ]:





# In[12]:


nomi_studenti = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
punteggi = [85, 92, 78, 88, 95]

plt.barh(nomi_studenti, punteggi, color='lightgreen')
plt.title('Punteggi degli Studenti')
plt.xlabel('Punteggio')
plt.ylabel('Nome dello Studente')
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd

nomi_studenti = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
punteggi = [85, 92, 78, 88, 95]

# Crea un DataFrame con nomi e punteggi
data = {'Nome dello Studente': nomi_studenti, 'Punteggio': punteggi}
df = pd.DataFrame(data)
# Ordina il DataFrame per punteggio in ordine crescente
df.sort_values(by='Punteggio', inplace=True)
df


# In[4]:


plt.barh(df['Nome dello Studente'], df['Punteggio'], color='lightgreen')
plt.title('Punteggi degli Studenti')
plt.xlabel('Punteggio')
plt.ylabel('Nome dello Studente')
plt.show()


# In[ ]:





# In[12]:


import matplotlib.pyplot as plt
import pandas as pd

nomi_studenti = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
punteggi = [85, 92, 78, 88, 95]

# Crea un DataFrame con nomi e punteggi
data = {'Nome dello Studente': nomi_studenti, 'Punteggio': punteggi}
df = pd.DataFrame(data)

# Ordina il DataFrame per punteggio in ordine crescente
df1=df.sort_values(by='Punteggio', inplace=False)


plt.barh(df1['Nome dello Studente'], df1['Punteggio'], color='lightgreen')
plt.title('Punteggi degli Studenti')
plt.xlabel('Punteggio')
plt.ylabel('Nome dello Studente')
plt.show()


# In[ ]:





# In[17]:


import matplotlib.pyplot as plt
import numpy as np
# Dati di esempio
altezza_maschi = np.random.normal(175, 10, 50)  # Altezza dei maschi
peso_maschi = np.random.normal(70, 5, 50)  # Peso dei maschi

altezza_femmine = np.random.normal(162, 8, 50)  # Altezza delle femmine
peso_femmine = np.random.normal(58, 4, 50)  # Peso delle femmine
# Crea il grafico a dispersione per i maschi
plt.scatter(altezza_maschi, peso_maschi, color='blue', label='Maschi', marker='o')

# Crea il grafico a dispersione per le femmine
plt.scatter(altezza_femmine, peso_femmine, color='pink', label='Femmine', marker='s')

# Personalizza il grafico
plt.title('Grafico a Dispersione Altezza vs Peso')
plt.xlabel('Altezza (cm)')
plt.ylabel('Peso (kg)')
plt.legend(loc='upper right')
plt.grid(True)

# Mostra il grafico
plt.show()


# In[ ]:





# In[ ]:





# In[5]:


# Dati di esempio
materie = ['Matematica', 'Scienze', 'Inglese', 'Storia', 'Arte']
maschi = [30, 25, 35, 20, 15]  # Numero di studenti maschi per materia
femmine = [20, 30, 25, 15, 10]  # Numero di studentesse per materia
# Creare il grafico a barre apilato
plt.figure(figsize=(10, 6))  # Imposta le dimensioni del grafico
plt.bar(materie, maschi, label='Maschi', color='blue')
plt.bar(materie, femmine, label='Femmine', bottom=maschi, color='lightpink')

# Personalizzare il grafico
plt.title('Grafico a Barre Apilato per Materia e Sesso')
plt.xlabel('Materie')
plt.ylabel('Numero di Studenti')
plt.legend(loc='upper right')

# Mostra il grafico
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


annata = ['Anno1', 'Anno2', 'Anno3']
gruppo1 = [90, 85, 88]
gruppo2 = [78, 92, 80]
gruppo3 = [85, 79, 91]
plt.plot(annata, gruppo1, marker='o', label='Gruppo 1', linestyle='-', color='blue')
plt.plot(annata, gruppo2, marker='s', label='Gruppo 2', linestyle='--', color='green')
plt.plot(annata, gruppo3, marker='^', label='Gruppo 3', linestyle='-.', color='red')

plt.title('Grafico a Linee delle Prestazioni per Annata')
plt.xlabel('Anno')
plt.ylabel('Prestazioni')
plt.legend(loc='upper left')

plt.show()


# In[ ]:





# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# Dati di esempio
annata = ['2020', '2021', '2022', '2023']
gruppo1 = [30, 40, 35, 50]
gruppo2 = [25, 35, 30, 45]
gruppo3 = [20, 30, 25, 40]


larghezza_barre = 0.1
indici = np.arange(len(annata))

plt.bar(indici - larghezza_barre, gruppo1, width=larghezza_barre, label='Gruppo 1', color='lightblue')
plt.bar(indici, gruppo2, width=larghezza_barre, label='Gruppo 2', color='lightgreen')
plt.bar(indici + larghezza_barre, gruppo3, width=larghezza_barre, label='Gruppo 3', color='brown')

plt.title('Grafico a Barre Raggruppate delle Prestazioni per Annata')
plt.xlabel('Anno')
plt.ylabel('Prestazioni')
plt.xticks(indici, annata)
plt.legend(loc='upper left')

plt.show()



# In[4]:


indici = np.arange(len(annata))


# In[6]:


indici - larghezza_barre


# In[26]:


import matplotlib.pyplot as plt
import numpy as np

# Passo 2: Crea dati di esempio
mesi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
vendite_prodotto_A = [100, 120, 90, 110, 95]
vendite_prodotto_B = [80, 95, 110, 85, 120]
vendite_prodotto_C = [60, 75, 70, 80, 65]

# Passo 3: Crea un grafico a barre empilato
plt.bar(mesi, vendite_prodotto_A, label='Prodotto A', color='blue')
plt.bar(mesi, vendite_prodotto_B, label='Prodotto B', color='green', bottom=vendite_prodotto_A)
plt.bar(mesi, vendite_prodotto_C, label='Prodotto C', color='red', bottom=np.array(vendite_prodotto_A) + np.array(vendite_prodotto_B))

# Passo 4: Personalizza il grafico
plt.title('Vendite Mensili per Prodotto')
plt.xlabel('Mese')
plt.ylabel('Vendite')
plt.legend(loc='upper left')

# Passo 5: Mostra il grafico risultante
plt.show()


# In[ ]:





# In[112]:


import matplotlib.pyplot as plt
import numpy as np

# Passo 2: Crea dati di esempio
città = ['Roma', 'Milano', 'Napoli', 'Torino', 'Firenze']
popolazione = [2870433, 1366180, 972198, 883767, 382258]

# Passo 3: Crea un grafico a barre orizzontali
plt.barh(città, popolazione, color='purple')
plt.title('Popolazione delle Città Italiane')
plt.xlabel('Popolazione')
plt.ylabel('Città')

# Passo 4: Mostra il grafico risultante
plt.show()


# In[ ]:





# In[27]:


import matplotlib.pyplot as plt
import numpy as np

# Passo 2: Crea dati di esempio
mesi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
vendite = [100, 120, 90, 110, 95]

# Passo 3: Crea un grafico a linee
plt.plot(mesi, vendite, marker='o', linestyle='-', color='blue')

# Passo 4: Personalizza il grafico
plt.title('Vendite Mensili di un Prodotto')
plt.xlabel('Mesi')
plt.ylabel('Vendite')
plt.grid(True)

# Passo 5: Mostra il grafico risultante
plt.show()


# In[ ]:





# In[ ]:





# In[29]:


import matplotlib.pyplot as plt

# Passo 2: Crea dati di esempio
attività = ['Lavoro', 'Studio', 'Tempo Libero']
percentuali = [50, 30, 20]
colori = ['gold', 'lightcoral', 'lightskyblue']

# Passo 3: Crea un grafico a torta
plt.pie(percentuali, labels=attività, colors=colori, autopct='%1.1f%%')

# Passo 4: Personalizza il grafico
plt.title('Utilizzo del Tempo')
plt.axis('equal')  # Rendi il grafico a torta circolare

# Passo 5: Mostra il grafico risultante
plt.show()


# In[ ]:





# In[35]:


import matplotlib.pyplot as plt
import numpy as np

# Passo 2: Crea dati di esempio
punteggi_matematica = [85, 92, 78, 88, 95, 90, 89, 86, 79, 91, 84, 87, 83, 82, 81, 80, 93, 94, 96, 97,85, 92, 78, 88, 95, 90, 89, 86, 79, 91, 84, 87, 83, 82, 81, 80, 93, 94, 96, 97]

punteggi_scienze = [78, 88, 85, 92, 90, 89, 86, 79, 91, 84, 87, 83, 82, 81, 80, 93, 94, 96, 97, 75,78, 88, 85, 92, 90, 89, 86, 79, 91, 84, 87, 83, 82, 81, 80, 93, 94, 96, 97, 75]



# Passo 3: Crea un grafico a dispersione
plt.scatter(punteggi_matematica, punteggi_scienze, color='purple', marker='o')

# Passo 4: Personalizza il grafico
plt.title('Punteggi di Matematica vs Scienze')
plt.xlabel('Punteggi di Matematica')
plt.ylabel('Punteggi di Scienze')
plt.grid(True)

# Passo 5: Mostra il grafico risultante
plt.show()


# In[36]:


import random

punteggi_matematica = []
# Set a length of the list to 10
for i in range(0, 50):
    punteggi_matematica.append(random.randint(70, 100))
    
    
    

punteggi_scienze = []
# Set a length of the list to 10
for i in range(0, 50):
    # any random numbers from 0 to 1000
    punteggi_scienze.append(random.randint(70, 100))
    


# Passo 3: Crea un grafico a dispersione
plt.scatter(punteggi_matematica, punteggi_scienze, color='purple', marker='o')

# Passo 4: Personalizza il grafico
plt.title('Punteggi di Matematica vs Scienze')
plt.xlabel('Punteggi di Matematica')
plt.ylabel('Punteggi di Scienze')
plt.grid(True)

# Passo 5: Mostra il grafico risultante
plt.show()



# In[ ]:





# In[ ]:





# In[121]:


import matplotlib.pyplot as plt

mesi = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio']
vendite = [12000, 15000, 18000, 20000, 22000]

plt.plot(mesi, vendite, marker='o', linestyle='-', color='blue')
plt.title('Andamento delle Vendite Mensili')
plt.xlabel('Mesi')
plt.ylabel('Vendite ($)')
plt.grid(True)
plt.show()


# In[122]:


import matplotlib.pyplot as plt
import numpy as np

categorie = ['Alimentari', 'Trasporti', 'Intrattenimento']
spese_anno_1 = [4000, 2500, 1500]
spese_anno_2 = [4200, 2700, 1600]

larghezza_barre = 0.35
indici = np.arange(len(categorie))

plt.bar(indici - larghezza_barre/2, spese_anno_1, width=larghezza_barre, label='Anno 1', color='blue')
plt.bar(indici + larghezza_barre/2, spese_anno_2, width=larghezza_barre, label='Anno 2', color='green')

plt.title('Spese Annuali per Categoria')
plt.xlabel('Categorie')
plt.ylabel('Spese ($)')
plt.xticks(indici, categorie)
plt.legend(loc='upper right')
plt.show()


# In[126]:


import matplotlib.pyplot as plt
import numpy as np

punteggi_matematica = [85, 92, 78, 88, 95]
punteggi_scienze = [78, 88, 85, 92, 90]

plt.scatter(punteggi_matematica, punteggi_scienze, color='purple', marker='o')
plt.title('Punteggi di Matematica vs Scienze')
plt.xlabel('Punteggi di Matematica')
plt.ylabel('Punteggi di Scienze')
plt.grid(True)
plt.show()


# In[128]:


import matplotlib.pyplot as plt

anni = [2010, 2011, 2012, 2013, 2014, 2015]
temperatura_media = [15, 16, 16.5, 17, 16.2, 15.8]

plt.plot(anni, temperatura_media, marker='o', linestyle='-', color='blue')
plt.title('Temperatura Annuale Media')
plt.xlabel('Anno')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.show()


# In[131]:


import matplotlib.pyplot as plt
import numpy as np

età = ['0-18', '19-35', '36-50', '51-65', '66+']
popolazione_maschile = [1000, 2500, 1800, 1200, 800]
popolazione_femminile = [950, 2400, 1700, 1100, 850]

indici = np.arange(len(età))

plt.bar(età, popolazione_maschile, label='Maschi', color='blue')
plt.bar(età, popolazione_femminile, label='Femmine', bottom=popolazione_maschile, color='pink')

plt.title('Distribuzione della Popolazione per Età e Genere')
plt.xlabel('Età')
plt.ylabel('Popolazione')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





# In[ ]:





# In[37]:


import matplotlib.pyplot as plt
import numpy as np

fasce_eta = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
popolazione_maschile = [1550000, 1600000, 1650000, 1700000, 1750000, 1780000, 1755000, 1720000, 1675000, 1600000]
popolazione_femminile = [1480000, 1550000, 1605000, 1650000, 1705000, 1750000, 1760000, 1725000, 1680000, 1595000]

indici = np.arange(len(fasce_eta))
larghezza_barre = 0.4

fig, ax1 = plt.subplots()

ax1.barh(indici, popolazione_maschile, height=larghezza_barre, label='Uomini', color='blue')
ax1.barh(indici, -popolazione_femminile, height=larghezza_barre, label='Donne', color='pink')

ax1.set_xlabel('Popolazione')
ax1.set_ylabel('Fasce d\'Età')
ax1.set_title('Piramide Demografica Italiana per Sesso e Età')
ax1.set_yticks(indici)
ax1.set_yticklabels(fasce_eta)
ax1.legend(loc='upper right')
ax1.invert_yaxis()  # Inverti l'asse y per rendere la piramide demografica

plt.show()




# In[136]:


import matplotlib.pyplot as plt
import numpy as np

fasce_eta = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
popolazione_maschile = [1550000, 1600000, 1650000, 1700000, 1750000, 1780000, 1755000, 1720000, 1675000, 1600000]
popolazione_femminile = [1480000, 1550000, 1605000, 1650000, 1705000, 1750000, 1760000, 1725000, 1680000, 1595000]

indici = np.arange(len(fasce_eta))
larghezza_barre = 0.4

fig, ax1 = plt.subplots()

ax1.barh(indici, popolazione_maschile, height=larghezza_barre, label='Uomini', color='blue')
ax1.barh(indici, [-x for x in popolazione_femminile], height=larghezza_barre, label='Donne', color='pink')

ax1.set_xlabel('Popolazione')
ax1.set_ylabel('Fasce d\'Età')
ax1.set_title('Piramide Demografica Italiana per Sesso e Età')
ax1.set_yticks(indici)
ax1.set_yticklabels(fasce_eta)
ax1.legend(loc='upper right')
ax1.invert_yaxis()  # Inverti l'asse y per rendere la piramide demografica

plt.show()


# In[43]:


import matplotlib.pyplot as plt
import numpy as np

fasce_eta = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
popolazione_maschile = [100000, 110000, 120000, 130000, 140000, 130000, 120000, 110000, 100000, 90000]
popolazione_femminile = [95000, 105000, 115000, 125000, 135000, 125000, 115000, 105000, 95000, 85000]

indici = np.arange(len(fasce_eta))
larghezza_barre = 0.4

fig, ax1 = plt.subplots()

ax1.barh(indici, popolazione_maschile, height=larghezza_barre, label='Uomini', color='blue')
ax1.barh(indici, [-x for x in popolazione_femminile], height=larghezza_barre, label='Donne', color='pink')

ax1.set_xlabel('Popolazione')
ax1.set_ylabel('Fasce d\'Età')
ax1.set_title('Piramide Demografica Italiana per Sesso e Età (Dati Fittizi)')
ax1.set_yticks(indici)
ax1.set_yticklabels(fasce_eta)
ax1.legend(loc='upper right')
#ax1.invert_yaxis()  # Inverti l'asse y per rendere la piramide demografica

plt.show()



# In[50]:


fasce_eta = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
             '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']

# Popolazione totale italiana (in milioni)
popolazione_totale = 60

# Stima realistica della popolazione maschile per ogni fascia di età (dati ipotetici)
percentuali_maschili = [49.5, 49.0, 48.5, 48.0, 47.5, 47.0, 46.5, 46.0, 45.5, 45.0,
                        44.5, 44.0, 43.5, 43.0, 42.5, 42.0, 41.5, 41.0, 40.5, 40.0, 30.0]

popolazione_maschile = []
popolazione_femminile = []

for percentuale in percentuali_maschili:
    popolazione_maschile.append(round((percentuale / 100) * (popolazione_totale * 1000000)))

# Calcolo della popolazione femminile per ogni fascia di età
for maschile in popolazione_maschile:
    popolazione_femminile = [(popolazione_totale * 1000000) - maschile]

indici = np.arange(len(fasce_eta))
larghezza_barre = 0.4

fig, ax1 = plt.subplots()

ax1.barh(indici, popolazione_maschile, height=larghezza_barre, label='Uomini', color='blue')
ax1.barh(indici, [-x for x in popolazione_femminile], height=larghezza_barre, label='Donne', color='pink')

ax1.set_xlabel('Popolazione')
ax1.set_ylabel('Fasce d\'Età')
ax1.set_title('Piramide Demografica Italiana per Sesso e Età (Dati Fittizi)')
ax1.set_yticks(indici)
ax1.set_yticklabels(fasce_eta)
ax1.legend(loc='upper right')
#ax1.invert_yaxis()  # Inverti l'asse y per rendere la piramide demografica

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





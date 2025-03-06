# Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva song.txt .
# Potrebno je napraviti rjeˇcnik koji kao kljuˇceve koristi sve razliˇcite rijeˇci koje se pojavljuju u
# datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijeˇc (kljuˇc) pojavljuje u datoteci.
# Koliko je rijeˇci koje se pojavljuju samo jednom u datoteci? Ispišite ih

wordDictionary = {}
songFile = open("song.txt")
for line in songFile:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word in wordDictionary.keys():
            wordDictionary[word] += 1
        else:
            wordDictionary[word] = 1
songFile.close()
for key,value in wordDictionary.items():
    if value == 1:
        print(key)
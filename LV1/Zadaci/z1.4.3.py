# Napišite program koji od korisnika zahtijeva unos brojeva u beskonaˇcnoj petlji
# sve dok korisnik ne upiše „ Done “ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
# potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
# vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
# (npr. slovo umjesto brojke) na naˇcin da program zanemari taj unos i ispiše odgovaraju´cu poruku.

userInputting = True
userInputList = []

while(userInputting):
    userInput = input()
    if userInput == "Done":
        userInputting = False
        break
    try:
        userInput = int(userInput)
        userInputList.append(userInput)
    except ValueError:
        print("Input contained letter, its skipped")
        continue

print(len(userInputList))
userInputList.sort()
print(min(userInputList))
print(max(userInputList))
print(sum(userInputList)/len(userInputList))
print(userInputList)
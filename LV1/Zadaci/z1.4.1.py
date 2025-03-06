#Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je pla´cen
#po radnom satu. Koristite ugra ¯denu Python metodu input() . Nakon toga izraˇcunajte koliko
#je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na naˇcin da ukupni iznos
#izraˇcunavate u zasebnoj funkciji naziva total_euro.

def total_euro(radniSati,vrijednostSata):
    return (radniSati * vrijednostSata)

radniSati = input("Unesite radne sate\n")
vrijednostSata = input("Unesite vrijednost sata\n")
print(f"{total_euro(float(radniSati),float(vrijednostSata))} eura" )



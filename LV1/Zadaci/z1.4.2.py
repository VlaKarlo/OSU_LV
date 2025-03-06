# Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
# nekakvu ocjenu i nalazi se izme ¯du 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju
# sljede´cih uvjeta:
# >= 0.9 A
# >= 0.8 B
# >= 0.7 C
# >= 0.6 D
# < 0.6 F
# Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
# Tako ¯der, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovaraju´cu poruku.

def printGrade():
    grade = "F"

    if ocjena >= 0.9:
        grade = "A"
    elif ocjena >= 0.8:
        grade = "B"
    elif ocjena >= 0.7:
        grade = "C"
    elif ocjena >= 0.6:
        grade = "D"

    print(grade)


try:
    ocjena = input()
    if ocjena == "":
        raise Exception("Grade is empty")
    ocjena = float(ocjena)
    if ocjena > 1.0 or ocjena < 0:
        raise Exception("Grade out of bounds")
    else:
        printGrade()

except Exception as e:
    print(f"{e}")
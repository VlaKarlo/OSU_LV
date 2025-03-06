# Napišite Python skriptu koja ´ce uˇcitati tekstualnu datoteku naziva SMSSpamCollection.txt
# [1]. Ova datoteka sadrži 5574 SMS poruka pri ˇcemu su neke oznaˇcene kao spam, a neke kao ham.
# Primjer dijela datoteke:
# ham Yup next stop.
# ham Ok lar... Joking wif u oni...
# spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
# a) Izraˇcunajte koliki je prosjeˇcan broj rijeˇci u SMS porukama koje su tipa ham, a koliko je
# prosjeˇcan broj rijeˇci u porukama koje su tipa spam.
# b) Koliko SMS poruka koje su tipa spam završava uskliˇcnikom ?

spamWithQuestionMark = 0
spamMessage = 0
spamMessageWords = 0
hamMessage = 0
hamMessageWords = 0
SMSFile = open("SMSSpamCollection.txt")
for line in SMSFile:
    line = line.rstrip()
    words = line.split()
    if words[0] == "spam" and words[-1][-1] == "?":
        spamWithQuestionMark += 1
    if words[0] == "spam":
        spamMessage += 1
        spamMessageWords += len(words)
    if words[0] == "ham":
        hamMessage += 1
        hamMessageWords += len(words)
SMSFile.close()
print(spamMessageWords/spamMessage)
print(hamMessageWords/hamMessage)
print(spamWithQuestionMark)
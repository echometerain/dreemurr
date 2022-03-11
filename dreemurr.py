from big_sleep import Imagine

file = open('lyrics.txt', 'r')
while True:
    line = file.readline()
    if not line:
        break
    dream = Imagine(
        text = line,
        lr = 5e-2,
        save_every = 25,
        save_progress = True
    )
    dream()
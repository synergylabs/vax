from os.path import expanduser
import datetime

file = open(expanduser("~") + '/Desktop/HERE.txt', 'w')
file.write("It worked!\n" + str(datetime.datetime.now()))
file.close()

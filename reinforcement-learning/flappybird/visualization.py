import matplotlib.pyplot as plt
import os,sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = BASE_DIR + "/process_record_s.txt"

y_value = list()

i = 0
with open(file_path,"r") as f:
    result = f.readline()
    result.strip()
    for item in result.split(' '):
        i+=1
        if item.isdigit():
            y_value.append(int(item))


print (i)
plt.plot(y_value)
plt.ylabel("living steps")
plt.show()

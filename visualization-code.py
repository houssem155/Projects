

import matplotlib.pyplot as plt
x=['java','python','PHP','JavaScript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
plt.bar(x_pos,popularity,color='blue')
plt.xlabel("lunguages")
plt.ylabel("popularity")
plt.title("popularity of programming lungage \n "+"worldwide,oct 2017 comared to a year ago")
plt.xticks(x_pos, x)
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
plt.show()

import matplotlib.pyplot as plt
x=['java','python','PHP','JavaScript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
plt.barh(x_pos,popularity,color='blue')
plt.xlabel("lunguages")
plt.ylabel("popularity")
plt.title("popularity of programming lungage \n "+"worldwide,oct 2017 comared to a year ago")
plt.yticks(x_pos, x)
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
plt.show()

import matplotlib.pyplot as plt
x=['java','python','PHP','JavaScript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
plt.bar(x_pos,popularity,color=['blue','red','green','yellow','cyan'])
plt.xlabel("lunguages")
plt.ylabel("popularity")
plt.title("popularity of programming lungage \n "+"worldwide,oct 2017 comared to a year ago")
plt.xticks(x_pos, x)
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
plt.show()

import matplotlib.pyplot as plt
x=['java','python','PHP','JavaScript','c#','c++']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i,_ in enumerate(x)]
fig, ax =plt.subplots()
rects1=ax.bar(x_pos,popularity, color='b')
plt.bar(x_pos,popularity,color='blue')
plt.xlabel("lunguages")
plt.ylabel("popularity")
plt.title("popularity of programming lungage \n "+"worldwide,oct 2017 comared to a year ago")
plt.xticks(x_pos, x)
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
def autolabel (rects):
  for rect in rects :
    height = rect.get_height()
    ax.text(rect.get_x()+ rect.get_width()/2. , 1*height, '%f' % float(height), ha='center' , va='bottom')
autolabel(rects1)
plt.show()

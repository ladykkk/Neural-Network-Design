from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import OrderedDict
import numpy as np

# tuples are species, age, weight in pounds
pet_list = [
    ("Cat", 5, 12),
    ("Rabbit", 1, 1.5),
    ("Dog", 1, 30),
    ("Cat", 7, 5),
    ("Rabbit", 2, 2.5),
    ("Dog", 5, 25),
    ("Bird", 23, .5),
    ("Rabbit", 1, 1.7),
    ("Dog", 2, 12),
    ("Cat", 3, 7),
    ("Rabbit", 4, 2.3),
    ("Dog", 12, 47),
    ("Cat", 12, 10),
    ("Bird", 13, .1),
    ("Dog", 7, 31),
    ("Cat", 11, 9),
    ("Rabbit", 2, 2.5),
    ("Cat", 1, 8),
    ("Bird", 11, .2),
    ("Cat", 2, 5),
    ("Bird", 3, .6),
    ("Rabbit", 2, 4.9),
    ("Cat", 7, 7),
    ("Bird", 7, .3),
    ("Rabbit", 1, 1.5)
]

pet_dict = OrderedDict()
for item in pet_list:
    pet_dict[item[0]] = pet_dict.get(item[0], 0) + 1
labels = pet_dict.keys()
sizes = pet_dict.values()
axes = [None]
figure, axes[0] = plt.subplots()


@pyqtSlot()
def draw_pie_chart():
    axes[0].remove()
    axes[0] = plt.axes()
    colors = ['orange', 'yellow', 'green', 'lightskyblue']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plot_space.draw_idle()


@pyqtSlot()
def draw_histogram():
    axes[0].remove()
    axes[0] = plt.axes()
    ages = [item[1] for item in pet_list]
    bins = np.arange(0.5, 100.5, 5)
    axes[0].hist(ages, bins=bins)
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Number of Pets")
    axes[0].set_xticks(bins)
    axes[0].set_xticklabels(bins)
    axes[0].set_xlim([min(ages) - 5, max(ages) + 5])
    plot_space.draw_idle()

my_app = QApplication([])
layout = QGridLayout()
win = QWidget()
win.setLayout(layout)

top_label = QLabel('Hello Foothill!')
top_label.setAlignment(Qt.AlignCenter)
second_label = QLabel('What plot would you like to see?')
second_label.setAlignment(Qt.AlignCenter)
pie_chart_response = QPushButton('Pie Chart')
histogram_response = QPushButton('Histogram')
plot_space = FigureCanvas(figure)

layout.addWidget(top_label, 0, 0, 1, 2)
layout.addWidget(second_label, 1, 0, 1, 2)
layout.addWidget(pie_chart_response, 2, 0, 1, 1)
layout.addWidget(histogram_response, 2, 1, 1, 1)
layout.addWidget(plot_space, 3, 0, 1, 2)

pie_chart_response.clicked.connect(draw_pie_chart)
histogram_response.clicked.connect(draw_histogram)

win.show()
my_app.exec()
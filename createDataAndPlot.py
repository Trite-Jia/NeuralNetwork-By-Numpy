import random
import numpy as np
import matplotlib.pyplot as plt

def create_data(data_num):
	datas = []
	for i in range(data_num):
		x = random.uniform(-2, 2)
		y = random.uniform(-2, 2)
		data = [x, y, tag_data(x, y)]
		datas.append(data)
	return np.array(datas)

def tag_data(x, y):
	return 1 if x**2 + y**2 < 1 else 0

def plot_data(datas, title):
	# 1 代表在圆内，为红色
	color = ["red" if i == 1 else 'blue' for i in datas[:, 2]]
	plt.scatter(datas[:, 0], datas[:, 1], c=color)
	plt.title(title)
	plt.show()

if __name__ == "__main__":
	datas = create_data(1000)
	print(datas)
	plot_data(datas, "red sun")
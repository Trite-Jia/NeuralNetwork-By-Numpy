import numpy as np
import createDataAndPlot as cp

# 权重矩阵，行数为输入数据的特征数，列数为要输出的神经元的个数
def create_weights(n_inputs, n_neurons):
	# 随机生成的权重矩阵
	weights = np.random.randn(n_inputs, n_neurons)
	return weights

# 偏置矩阵，在每个输出神经元之后添加上一个数
def create_biases(n_neurons):
	biases = np.random.randn(n_neurons)
	return biases

# 激活函数，ReLU
def activation_function(x):
	return np.maximum(0, x)

# 激活函数，归一化指数函数，用于神经网络的最后一层，最后一层不使用 ReLU
def softmax(x):
	# axis=1 代表取每行的最大值，keepdims=True 代表保持矩阵结构，此处代表结果为列向量
	# 只传入一条数据的话，此处 axis=1 会报错

	# 先让输入平移到负数区段，保证指数值不溢出
	# 使用指数值保证所有数字都是正数，计算概率时可以让分母去除负数与零的情况
	exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
	return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# 标准函数，为避免数过大或者过小，对数据按比例调整
def normalize(x):
	# 调整过后的一条数据的最大绝对值为1（数据全为0除外）
	# 1,4 => 0.25 1
	# -1, 4 => -0.25, 1
	max_value = np.max(np.abs(x), axis=1, keepdims=True)
	# np.where(cond, a, b) 的执行顺序是先算 a 和 b，再按 cond 选择。
	# numpy 重写了向量的 != 表达式，对数组中每个元素分别与 0 比较，返回同形状的布尔数组
	return x / np.where(max_value != 0, max_value, 1)

# 神经网络的一层
class Layer:
	def __init__(self, n_inputs, n_neurons):
		self.weights = np.random.randn(n_inputs, n_neurons)
		self.biases = np.random.randn(n_neurons)
	def forward(self, inputs):
		# self.biases 的矩阵格式可能并不匹配，但 numpy 还是保证可以进行计算
		# 计算这个神经元的时候，便让 ReLU 有了两个参数，ReLU(wx + b)，可以更改激活函数
		# 的斜率与位置
		return np.dot(inputs, self.weights) + self.biases


class Network:
	def __init__(self, network_shape):
		self.layers = []
		self.network_shape = network_shape
		for i in range(len(network_shape) - 1):
			layer = Layer(network_shape[i], network_shape[i + 1])
			self.layers.append(layer)

	def forward(self, inputs):
		length = len(self.layers)
		for i in range(length):
			layer = self.layers[i]
			inputs = layer.forward(inputs)
			# 进行激活
			if i != length - 1:
				inputs = activation_function(inputs)
				# 标准化
				inputs = normalize(inputs)
			else:
				inputs = softmax(inputs)
		return inputs


def classify(inputs):
	# 对第二列四舍五入，较小的话则为0，较大的话则为1，
	# 此时模型预测其在圆内
	return np.rint(inputs[:,1])

if __name__ == "__main__":
	# 得到标记之后的数据
	data = cp.create_data(200)
	cp.plot_data(data, " ")
	inputs = data[:, (0, 1)]	# 取前两列的坐标
	network = Network([2, 3, 4, 2])
	outputs = network.forward(inputs)
	data[:, 2] = classify(outputs)
	cp.plot_data(data, " ")
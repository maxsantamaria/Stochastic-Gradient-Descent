import numpy as np


def prediction(x, w):
	return np.matmul(x, w)


def loss_function(x, y, w, lambd=0):
	n = x.shape[0]
	loss = 1/n * np.matmul(np.transpose(y - np.matmul(x, w)), y - np.matmul(x, w)) + lambd * np.matmul(w.T, w)
	return loss[0][0]  # loss is a 1x1 matrix


def derivative(x, y, w, lambd=0):
	d = -2/n * np.matmul(x.T, y - np.matmul(x, w)) + lambd * 2 * w
	return d  # (k+1)x1 matrix


def SGD(x, y, alpha, lambd, nepoch, epsilon, w):
	iterations = 100
	n = x.shape[0]
	k = x.shape[1] - 1
	loss_history = []
	w_history = []
	for i in range(iterations):
		random_num = np.random.randint(0, n)
		x_i = x[random_num, :].reshape(1, k + 1)
		y_i = y[random_num].reshape(1, 1)
		w = w - alpha * derivative(x, y, w, lambd)
		loss = loss_function(x, y, w, lambd)
		loss_history.append(loss)
		w_history.append(w)
		if loss < epsilon:
			break
	print(loss_history)
	print(w)
	print(loss)
	#print(w_history)


def SGDSolver(x, y, alpha, lam, nepoch, epsilon, param):
	# Training Phase
	SGD(x, y, alpha, lam, nepoch, epsilon, param)


def generate_data(n, k, bias=True):
	x = np.random.randn(n, k)
	if bias:
		x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s
	y = np.random.randn(n, 1)
	return x, y


def reader(file, bias=True):
	with open(file, 'r') as data:
		keys = data.readline().replace(' ,', ',').strip().split(',')[1:]
		print(keys)
		matrix = None
		for line in data:
			row = line.strip().split(',')[1:]
			row = list(map(float, row))
			if matrix is None:
				matrix = np.array([row])
			else:
				matrix = np.append(matrix, [row], axis=0)
		n = matrix.shape[0]
		y = matrix[:,-1]
		y = y.reshape(n, 1)  # make it a matrix nx1
		x = matrix[:,:-1]
		if bias:
			x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s
		return x, y


if __name__ == "__main__":
	n = 10
	k = 1
	# x, y = generate_data(n, k)
	x, y = reader('Admission_Predict.csv')
	k = x.shape[1] - 1
	w = np.random.randn(k + 1, 1)
	SGDSolver(x, y, 0.00000001, 0.5, 0, 0.5, w)
	#SGD(x, y, 0.00000001, 0.5, 0, 0.5)
	


import numpy as np


def predict(x, w):
	return np.matmul(x, w)


def loss_function(x, y, w, lambd=0):
	n = x.shape[0]
	prediction = predict(x, w)
	loss = 1/n * np.matmul(np.transpose(y - prediction), y - prediction) + lambd * np.matmul(w.T, w)
	return loss[0][0]  # loss is a 1x1 matrix


def derivative(x, y, w, lambd=0):
	prediction = predict(x, w)
	d = -2/n * np.matmul(x.T, y - prediction) + lambd * 2 * w
	return d  # (k+1)x1 matrix


def SGD(x, y, alpha, lambd, nepoch, epsilon, w):
	iterations = 1000
	n = x.shape[0]
	k = x.shape[1] - 1
	loss_history = [loss_function(x, y, w, lambd)]
	w_history = []
	for i in range(iterations):
		random_num = np.random.randint(0, n)
		#x_i = x[random_num, :].reshape(1, k + 1)
		#y_i = y[random_num].reshape(1, 1)
		'''
		w = w - alpha * derivative(x, y, w, lambd)
		loss = loss_function(x, y, w, lambd)
		loss_history.append(loss)
		w_history.append(w)
		'''
		for i, row in enumerate(x):  # Each sample
			row = row.reshape(1, row.shape[0])
			prediction = predict(row, w).reshape(1, 1)
			w = w - alpha * ((-2/n) * row.T * (y[i] - prediction) + lambd * 2 * w)
		loss = loss_function(x, y, w, lambd)
		loss_history.append(loss)
		w_history.append(w)

		if loss < epsilon:
			break
	print(loss_history)
	print(w)
	print(loss)
	#print(w_history)
	return w


def GD(x, y, alpha, lambd, nepoch, epsilon, w):
	#iterations = 1000
	n = x.shape[0]
	k = x.shape[1] - 1
	loss_history = [loss_function(x, y, w, lambd)]
	w_history = []
	for i in range(nepoch):
		random_num = np.random.randint(0, n)
		w = w - alpha * derivative(x, y, w, lambd)
		loss = loss_function(x, y, w, lambd)
		loss_history.append(loss)
		w_history.append(w)
		if loss < epsilon:
			break
	#print(loss_history)
	print(w)
	print(loss)
	#print(w_history)


def error(x, y, w):
	n = x.shape[0]
	prediction = predict(x, w)
	error = y - prediction
	error_sum = 0
	for i in range(n):
		error_sum += error[i][0]**2
	print(error_sum/n)


def SGDSolver(x, y, alpha, lam, nepoch, epsilon, param):
	# Training Phase
	param = SGD(x, y, alpha, lam, nepoch, epsilon, param)
	# Validation Phase
	error(x, y, param)
	print(np.matmul(x[27], param))


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
	SGDSolver(x, y, 0.001, 0.5, 1000, 0.05, w)
	#GD(x, y, 0.0000001, 0.5, 1000, 0.5, w)
	#error(x, y, w)
	



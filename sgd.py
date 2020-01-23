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
		data = np.append(x, y, axis=1)
		np.random.shuffle(data)
		x = data[:,:k+1]
		y = data[:,k+1:]
		for i, row in enumerate(x):  # Each sample
			row = row.reshape(1, row.shape[0])
			prediction = predict(row, w).reshape(1, 1)
			w = w - alpha * ((-2/n) * row.T * (y[i] - prediction) + lambd * 2 * w)
		loss = loss_function(x, y, w, lambd)
		loss_history.append(loss)
		w_history.append(w)

		if loss < epsilon:
			break
	#print(loss_history)
	#print(w)
	#print(w_history)
	#print('FINAL LOSS: ', loss)

	return w, loss


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
	#print(w)
	#print('FINAL LOSS: ', loss)


def error(x, y, w):
	n = x.shape[0]
	prediction = predict(x, w)
	error = y - prediction
	error_sum = 0
	for i in range(n):
		error_sum += error[i][0]**2
	#print(error_sum/n)
	return error_sum/n


def SGDSolver(x, y=None, alpha=None, lam=None, nepoch=1000, epsilon=0.002, param=None):
	n = x.shape[0]
	x = normalize(x)
	x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s
	if alpha is not None and lam is not None:
		#print("Training phase")
		# Training Phase
		min_loss = 10**10
		best_param = param
		# for lr in np.arange(alpha[0], alpha[1], (alpha[1] - alpha[0]) / 5.0): # Linear Scale
		for lr in np.logspace(np.log10(alpha[0]), np.log10(alpha[1]), 5):  # Logaritmic Scale
			#print('\nNEW LEARNING RATE ', lr)
			# for regularization_weight in np.arange(lam[0], lam[1], (lam[1] - lam[0]) / 5.0):
			for regularization_weight in np.logspace(np.log10(lam[0]), np.log10(lam[1]), 5):
				#print('\tNEW LAMBDA ', regularization_weight)
				w, loss = SGD(x, y, lr, regularization_weight, nepoch, epsilon, param)
				if loss < min_loss:
					min_loss = loss
					best_param = w
					best_lr = lr
					best_lam = regularization_weight
		#print('Min LOSS', min_loss, 'Best LR', best_lr, 'Best Lambda', best_lam)
		#print(np.matmul(x[:28], best_param))
		return best_param
	elif alpha is None: #y is not None:
		# Testing Phase
		#print("Testing phase")
		param = y
		return predict(x, param)
	elif lam is None:
		# Validation Phase
		#print("Validation phase")
		param = alpha
		return error(x, y, param)	
	

def normalize(x):
	n = x.shape[0]
	k = x.shape[1]
	for i in range(k):
		feature = x[:, i]
		min_x = min(feature)
		max_x = max(feature)
		feature = (feature - min_x) / (max_x - min_x)
		feature = feature - np.mean(feature)
		x[:, i] = feature
	return x


def generate_data(n, k, bias=True):
	x = np.random.randn(n, k)
	if bias:
		x = np.hstack((np.array([1] * n)[:, np.newaxis], x))  # Add a column of 1s
	y = np.random.randn(n, 1)
	return x, y


def reader(file, bias=True):
	with open(file, 'r') as data:
		keys = data.readline().replace(' ,', ',').strip().split(',')[1:]
		#print(keys)
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
	x, y = reader('Admission_Predict.csv', bias=False)
	k = x.shape[1]
	w = np.random.randn(k + 1, 1)
	#w = np.array([1] * (k + 1)).reshape(k+1, 1)
	# best lr before=0.001, best lr now=0.0001-0.0002
	#w = SGDSolver(x, y, [0.2, 1], [0.000001, 0.00001], 1000, 0.002, w)  # error was 0.01
	w = SGDSolver(x, y, [0.1, 2], [0.0000001, 0.00001], 1000, 0.002, w)
	print(SGDSolver(x, y, w))
	print(SGDSolver(x, w))
	
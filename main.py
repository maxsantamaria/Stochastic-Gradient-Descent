import numpy as np


def reader(file):
	with open(file, 'r') as data:
		#print(data.readline())
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
		print(matrix)
		return matrix
			

def training(x, y, alpha, lambd, nepoch):
	print(x.shape[1])
	k = x.shape[1]
	n = x.shape[0]
	x = np.hstack((np.array([1] * n)[:, np.newaxis], x))
	w = np.array([0] * (k + 1))  # Initialize w
	matrix1 = y - np.matmul(x, w)
	loss_function = np.matmul(np.transpose(matrix1), matrix1) + lambd * np.dot(w, w)
	derivative = -2 * np.matmul(np.transpose(x), matrix1) + 2 * lambd * w
	iterations = 10
	loss_history = np.zeros(iterations)
	w_history = np.zeros((iterations, k + 1))
	for i in range(iterations):
		
		w = w - alpha * (-2 * np.matmul(np.transpose(x), matrix1) + 2 * lambd * w)
		matrix1 = y - np.matmul(x, w)
		loss_history[i] = np.matmul(np.transpose(matrix1), matrix1) + lambd * np.dot(w, w)
		w_history[i,:] = w.T
		
	print(w_history)
	print(loss_history)


def  cal_cost(theta,X,y):
    '''
    
    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas 
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))
    
    where:
        j is the no of features
    '''
    
    m = len(y)
    
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    X = np.hstack((np.array([1] * X.shape[0])[:, np.newaxis], X))

    m = y.shape[0]
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,X.shape[1]))
    for it in range(iterations):
        
        prediction = np.dot(X,theta)
        print(y.T.shape)
       	print(np.matmul(X.T, (prediction - y)).shape)

        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        print(theta.shape)

        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history





if __name__ == "__main__":


	array1 = np.array([1, 4, 5])
	array2 = np.array([1, 1, 1])
	matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
	matrix2 = np.array([[1, 1], [1, 1], [1, 1]])
	print(array1.shape)
	x = reader('Admission_Predict.csv')

	y = x[:,-1]
	x = x[:,:-1]

	training(x, y, 0.001, 0, 0)

	#print(np.matmul(matrix2, matrix1))
	#print(np.dot(array1, array2))
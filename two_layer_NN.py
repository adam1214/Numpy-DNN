# https://www.brilliantcode.net/1381/backpropagation-2-forward-pass-backward-pass/?cli_action=1623400223.646
# * 與 np.multiply : element-wise product
# @ 與 np.dot : 矩陣相乘
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import PCA
import glob

def load_feature_label(train_or_test, feature_list, label_list):
    if train_or_test == 0:
        target_dir_list = ['./Data_train/Carambula', './Data_train/Lychee', './Data_train/Pear']
    else:
        target_dir_list = ['./Data_test/Carambula', './Data_test/Lychee', './Data_test/Pear']
    
    for target_dir in target_dir_list:
        files = glob.glob(target_dir+'/*.png')
        for file in files:
            img = imageio.imread(file)  #(32,32,4)
            img = (img.flatten()).astype(np.float)  #(4096,) 
            img = img/255   #[0, 255] -> [0, 1]
            feature_list.append(img)
            if target_dir_list.index(target_dir) == 0:
                label_list.append([1,0,0])
            elif target_dir_list.index(target_dir) == 1:
                label_list.append([0,1,0])
            else:
                label_list.append([0,0,1])

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros([n_h, 1])
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros([n_y, 1])
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

def forwardpass(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m) (2,1)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    
    W1 = parameters["W1"] #(5,2)
    b1 = parameters["b1"] #(5,1)
    W2 = parameters["W2"] #(3,5)
    b2 = parameters["b2"] #(3,1)
    
    Z1 = W1@X + b1
    A1 = 1/(1+np.exp(-Z1))  #Sigmoid (5,1)
    Z2 = W2@A1 + b2
    A2 = np.exp(Z2)/np.sum(np.exp(Z2), axis=0)  #Softmax (3,1)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_cost(A2, Y, parameters):
    batch_size = Y.shape[0]   # batch size is 1 when training, because of SGD, Y.shape=(1,3)
    
    #cost = (1./m) * (-np.dot(Y,np.log(A2).T) - np.dot(1-Y, np.log(1-A2).T))
    cost = (1./batch_size) * -(Y @ np.log(A2))
    
    #cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    return cost

def backwardpass(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, 1)
    Y -- "true" labels vector of shape (3, 1)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    
    batch_size = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    dA2 = - np.divide(Y, A2) #(3,1)
    #dZ2 = dA2 * ( (np.exp(Z2)/np.sum(np.exp(Z2))) - (np.exp(2*Z2)/(np.sum(np.exp(Z2)))**2) ) #(3,1) # only consider i == j, no i != J
    dZ2 = A2 - Y # ref: https://deepnotes.io/softmax-crossentropy?fbclid=IwAR1KALP-_e5K5B2XVX0wy9q52FItJnICfcD2AxSmWcVOcnUkZyEFLgzypHk#derivative-of-softmax
    dW2 = (dZ2 @ A1.T)/batch_size #(3,5)
    db2 = dZ2/batch_size #(3,1)
    dA1 = W2.T @ dZ2 #(5,1)

    temp_s = 1/(1+np.exp(-Z1))
    dZ1 = dA1 * temp_s * (1-temp_s) #(5,1)
    dW1 = (dZ1 @ X.T)/batch_size #(5,2)
    db1 = dZ1/batch_size #(5,1)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

if __name__ == "__main__":
    np.random.seed(1)
    learning_rate = 0.02
    
    train_feature = []
    train_label = []
    load_feature_label(train_or_test = 0, feature_list = train_feature, label_list = train_label)
    train_label = np.array(train_label).astype(np.float)
    train_label_arr = (np.ones(train_label.shape[0])*-1).astype(np.int)
    for i in range(0, train_label.shape[0], 1):
        if np.array_equal(train_label[i], np.array([1,0,0])):
            train_label_arr[i] = 0
        elif np.array_equal(train_label[i], np.array([0,1,0])):
            train_label_arr[i] = 1
        elif np.array_equal(train_label[i], np.array([0,0,1])):
            train_label_arr[i] = 2

    test_feature = []
    test_label = []
    load_feature_label(train_or_test = 1, feature_list = test_feature, label_list = test_label)
    test_label = np.array(test_label)
    for i in range(0, test_label.shape[0], 1):
        if np.array_equal(test_label[i], np.array([1,0,0])):
            test_label[i] = 0
        elif np.array_equal(test_label[i], np.array([0,1,0])):
            test_label[i] = 1
        elif np.array_equal(test_label[i], np.array([0,0,1])):
            test_label[i] = 2
    test_label = test_label[:,0]    #(498,) np int array
    
    pca = PCA(n_components=2)
    train_feature_pca = pca.fit_transform(train_feature)    #(1470, 2) np float array
    test_feature_pca = pca.transform(test_feature)   #(498, 2) np float array

    parameters = initialize_parameters(n_x=2, n_h=35, n_y=3)
    rand_pick_idx = np.arange(train_feature_pca.shape[0]) 
    np.random.shuffle(rand_pick_idx) 
    
    train_acc_list = []
    test_acc_list = []
    #training
    for i in range(0, rand_pick_idx.shape[0], 1):
        A2, cache = forwardpass(train_feature_pca[rand_pick_idx[i]].reshape(2,1), parameters)
        cost = compute_cost(A2, train_label[rand_pick_idx[i]].reshape(1,3), parameters)
        #print(cost)
        grads = backwardpass(parameters, cache, train_feature_pca[rand_pick_idx[i]].reshape(2,1), train_label[rand_pick_idx[i]].reshape(3,1))
        parameters = update_parameters(parameters, grads, learning_rate)
        
        #predict training data
        probs, caches = forwardpass(train_feature_pca.T, parameters)
        predict_labels = np.argmax(probs, axis=0)
        acc = 0
        for a in range(0, predict_labels.shape[0], 1):
            if predict_labels[a] == train_label_arr[a]:
                acc += 1
        #print('model acc on training dataset:', acc/predict_labels.shape[0])
        train_acc_list.append(acc/predict_labels.shape[0])
        
        #predict testing data
        probs, caches = forwardpass(test_feature_pca.T, parameters) #probs:(3, 498)
        predict_test_labels = np.argmax(probs, axis=0)
        acc = 0
        for a in range(0, predict_test_labels.shape[0], 1):
            if predict_test_labels[a] == test_label[a]:
                acc += 1
        #print('model acc on testing dataset:', acc/predict_labels.shape[0])
        test_acc_list.append(acc/predict_test_labels.shape[0])

    print('model acc on testing dataset:', acc/predict_test_labels.shape[0])
    plt.figure()
    plt.plot(np.arange(train_feature_pca.shape[0]), train_acc_list, color = 'r', label="training data acc")
    plt.plot(np.arange(train_feature_pca.shape[0]), test_acc_list, color = 'b', label="testing data acc")
    plt.legend(loc='lower right')
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.show()
    #plt.save('train_test_acc.png')

    colors = ('green', 'blue', 'red')
    cmap = ListedColormap(colors) #上色器

    # feature value boundary
    x1_min = np.min(test_feature_pca[:, 0])
    x1_max = np.max(test_feature_pca[:, 0]) + 1
    x2_min = np.min(test_feature_pca[:, 1])
    x2_max = np.max(test_feature_pca[:, 1]) + 1

    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    x1_x2_flatten = np.array([x1_mesh.flatten(), x2_mesh.flatten()]).T #(3814834, 2)
    probs, caches = forwardpass(x1_x2_flatten.T, parameters) #probs:(3, 3814834)
    predict_labels = np.argmax(probs, axis=0)
    predict_labels = predict_labels.reshape(x1_mesh.shape)
    plt.figure()
    plt.contourf(x1_mesh, x2_mesh, predict_labels, cmap=cmap, alpha=0.5) #三維等高線圖
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    # plot testing data ground true
    type0 = []
    type1 = []
    type2 = []
    for i in range(0, test_feature_pca.shape[0], 1):
        if test_label[i] == 0:
            type0.append(test_feature_pca[i])
        elif test_label[i] == 1:
            type1.append(test_feature_pca[i])
        elif test_label[i] == 2:
            type2.append(test_feature_pca[i])
    type0_arr = np.array(type0)
    type1_arr = np.array(type1)
    type2_arr = np.array(type2)
    plt.scatter(x=type0_arr[:, 0], y=type0_arr[:, 1], c=colors[0])
    plt.scatter(x=type1_arr[:, 0], y=type1_arr[:, 1], c=colors[1])
    plt.scatter(x=type2_arr[:, 0], y=type2_arr[:, 1], c=colors[2])
    plt.show()
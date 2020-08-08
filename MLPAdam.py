from tensorflow.keras.layers import Input, Dense, Reshape, Softmax
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')

state_dim = 21+21+101
action_dim = 1
atoms = 1
batch_size = 10
#tau = np.array([0.1,0.3,0.5,0.7,0.9])
tau = np.array([0.1])

model1 = tf.keras.Sequential([
            Input([state_dim, ]),
            #Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            #Dense(32, activation='relu'),
            Dense(action_dim * atoms, activation='linear'),
            Reshape([action_dim, atoms])
        ])

state = np.array([[0,10,0],[1,11,50],[2,12,79],[3,11,54],[4,10,65],[5,11,44],[6,13,65],[7,12,32],[8,9,43],[9,10,75]])
true_y_mean = [100,20,-50,40,80,-130,-340,20,-100,-10]
N = 2000
true_y = np.zeros((N,10,1))
state_matrix = np.zeros((batch_size,21+21+101))
for i in range(len(state)):
    state_matrix[i][state[i][0]] = 1
    state_matrix[i][state[i][1]+21] = 1
    state_matrix[i][state[i][2]+21+21] = 1
    

def loss1(true_y,theta):
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(tf.expand_dims(true_y,-1),theta),axis=2),axis=1))

def lossHuber(true_y,theta,atoms,tau):
    true_y_value = tf.tile(tf.expand_dims(true_y,-1),[1,1,atoms])
    error = tf.math.subtract(true_y_value,theta)
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(tf.expand_dims((true_y_value),-1),tf.expand_dims(theta,-1))
    tau_value = tf.tile(tf.expand_dims(tf.expand_dims(tau,0),0),[10,1,1])
    inv_tau_value = tf.tile(tf.expand_dims(tf.expand_dims(1.0-tau,0),0),[10,1,1])
    loss= tf.where(tf.less(error, 0.0),inv_tau_value*huber,tau_value*huber)
    return tf.reduce_mean(tf.reduce_sum(loss,axis=-1))

def lossQR(true_y,theta,atoms,tau):
    true_y_value = tf.tile(tf.expand_dims(true_y,-1),[1,1,atoms])
    error = tf.math.subtract(true_y_value,theta)
    #huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(tf.expand_dims((true_y_value),-1),tf.expand_dims(theta,-1))
    tau_value = tf.tile(tf.expand_dims(tf.expand_dims(tau,0),0),[10,1,1])
    tau_minusone_value = tf.tile(tf.expand_dims(tf.expand_dims(tau-1.0,0),0),[10,1,1])
    loss= tf.where(tf.less(error, 0.0),tau_minusone_value*error,tau_value*error)
    return tf.reduce_mean(tf.reduce_sum(loss,axis=-1))

allloss = np.array([])
for i in range(N):
    for j in range(10):
        true_y[i][j][0] = 10*np.random.randn()+true_y_mean[j]
    with tf.GradientTape() as tape:
        theta = model1(state_matrix)
        loss = lossHuber(true_y[i],theta,atoms,tau)
        print("loss before adam is{}".format(loss))
        allloss = np.append(allloss,loss.numpy())
    gradients = tape.gradient(loss,model1.trainable_variables)
    tf.keras.optimizers.Adam(0.001).apply_gradients(zip(gradients, model1.trainable_variables))
    theta = model1(state_matrix)
    loss = lossHuber(true_y[i],theta,atoms,tau)
    print("loss after adam is{}".format(loss))
pred = model1.predict(state_matrix)
np.quantile(true_y,0.1,axis=0)
pred[:,:,0]
a = pred[:,:,0]-np.quantile(true_y,0.1,axis=0)

plt.figure(1, figsize=(12,8))
plt.grid()
plt.plot(allloss)
plt.ylabel('loss')
plt.xlabel("steps*100")
plt.title("LOSS",fontsize=25)

import numpy as np
from utils.Convolution_util   import zero_pad, conv_forward, conv_backward, conv_SDLM
from utils.Pooling_util       import pool_forward, pool_backward, subsampling_forward, subsampling_backward
from utils.Activation_util    import activation_func
from utils.RBF_initial_weight import rbf_init_weight
from utils.utils_func         import *

class ConvLayer(object):
    def __init__(self, kernel_shape, hparameters, init_mode='Gaussian_dist'):
        """
        kernel_shape: (n_f, n_f, n_C_prev, n_C)
        hparameters = {"stride": s, "pad": p}
        """
        self.hparameters = hparameters
        self.weight, self.bias = initialize(kernel_shape, init_mode)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        
    def foward_prop(self, input_map):
        output_map, self.cache = conv_forward(input_map, self.weight, self.bias, self.hparameters)
        return output_map
    
    def back_prop(self, dZ, momentum, weight_decay):
        dA_prev, dW, db = conv_backward(dZ, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev  
    
    def SDLM(self, d2Z, mu, lr_global):
        d2A_prev, d2W = conv_SDLM(d2Z, self.cache)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev  
    
# C3: Convlayer with assigned combination between input maps and weight
class ConvLayer_maps(object):
    def __init__(self, kernel_shape, hparameters, mapping, init_mode='Gaussian_dist'):
        """
        kernel_shape: (n_f, n_f, n_C_prev, n_C)
        hparameters = {"stride": s, "pad": p}
        """
        self.hparameters = hparameters
        self.mapping     = mapping
        self.wb   = []      # list of [weight, bias]
        self.v_wb = []      # list of [v_w,    v_b]
        for i in range(len(self.mapping)):
            weight_shape = (kernel_shape[0], kernel_shape[1], len(self.mapping[i]), 1)
            w, b = initialize(weight_shape, init_mode)
            self.wb.append([w, b])
            self.v_wb.append([np.zeros(w.shape), np.zeros(b.shape)])
        
    def foward_prop(self, input_map):
        self.iputmap_shape = input_map.shape #(n_m,14,14,6)
        self.caches = []
        output_maps = []
        for i in range(len(self.mapping)):
            output_map, cache = conv_forward(input_map[:,:,:,self.mapping[i]], self.wb[i][0], self.wb[i][1], self.hparameters)
            output_maps.append(output_map)
            self.caches.append(cache)
        output_maps = np.swapaxes(np.array(output_maps),0,4)[0]
        return output_maps
    
    def back_prop(self, dZ, momentum, weight_decay):
        dA_prevs = np.zeros(self.iputmap_shape)
        for i in range(len(self.mapping)):
            dA_prev, dW, db = conv_backward(dZ[:,:,:,i:i+1], self.caches[i])
            self.wb[i][0], self.wb[i][1], self.v_wb[i][0], self.v_wb[i][1] =\
                update(self.wb[i][0], self.wb[i][1], dW, db, self.v_wb[i][0], self.v_wb[i][1], self.lr, momentum, weight_decay)
            dA_prevs[:,:,:,self.mapping[i]] += dA_prev
        return dA_prevs 
    
    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2Z, mu, lr_global):
        h = 0
        d2A_prevs = np.zeros(self.iputmap_shape)
        for i in range(len(self.mapping)):
            d2A_prev, d2W = conv_SDLM(d2Z[:,:,:,i:i+1], self.caches[i])
            d2A_prevs[:,:,:,self.mapping[i]] += d2A_prev
            h += np.sum(d2W)
        self.lr = lr_global / (mu + h/d2Z.shape[0])
        return d2A_prevs 

class PoolingLayer(object):
    def __init__(self, hparameters, mode):
        self.hparameters = hparameters
        self.mode = mode
        
    def foward_prop(self, input_map):   # n,28,28,6 / n,10,10,16
        A, self.cache = pool_forward(input_map, self.hparameters, self.mode)
        return A
    
    def back_prop(self, dA):
        dA_prev = pool_backward(dA, self.cache, self.mode)
        return dA_prev
    
    def SDLM(self, d2A):
        d2A_prev = pool_backward(d2A, self.cache, self.mode)
        return d2A_prev

class Subsampling(object):
    def __init__(self, n_kernel, hparameters):
        self.hparameters = hparameters
        self.weight = np.random.normal(0, 0.1, (1,1,1,n_kernel)) 
        self.bias   = np.random.normal(0, 0.1, (1,1,1,n_kernel)) 
        self.v_w = np.zeros(self.weight.shape)
        self.v_b = np.zeros(self.bias.shape)
        
    def foward_prop(self, input_map):   # n,28,28,6 / n,10,10,16
        A, self.cache = subsampling_forward(input_map, self.weight, self.bias, self.hparameters)
        return A
    
    def back_prop(self, dA, momentum, weight_decay):
        dA_prev, dW, db = subsampling_backward(dA, A_, weight, b, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev
    
    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2A, mu, lr_global):
        d2A_prev, d2W, _ = subsampling_backward(dA, A_, weight, b, self.cache)
        h = np.sum(d2W)/d2A.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev

class Activation(object):
    def __init__(self, mode):    
        (act, d_act), actfName = activation_func()
        act_index  = actfName.index(mode)
        self.act   = act[act_index]
        self.d_act = d_act[act_index]
        
    def foward_prop(self, input_image): 
        self.input_image = input_image
        return self.act(input_image)
    
    def back_prop(self, dZ):
        dA = np.multiply(dZ, self.d_act(self.input_image)) 
        return dA
    
    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2Z):  #d2_LeNet5_squash
        dA = np.multiply(d2Z, np.power(self.d_act(self.input_image),2)) 
        return dA

class FCLayer(object):
    def __init__(self, weight_shape, init_mode='Gaussian_dist'): 
        
        # Initialization
        self.v_w, self.v_b = np.zeros(weight_shape), np.zeros((weight_shape[-1],))
        self.weight, self.bias = initialize(weight_shape, init_mode)
        
    def foward_prop(self, input_array):
        self.input_array = input_array  #(n_m, 120)
        return np.matmul(self.input_array, self.weight) # (n_m, 84)
        
    def back_prop(self, dZ, momentum, weight_decay):
        dA = np.matmul(dZ, self.weight.T)               # (n_m, 84) * (84, 120) = (n_m, 120)
        dW = np.matmul(self.input_array.T, dZ)          # (n_m, 120).T * (n_m, 84) = (120, 84)
        db = np.sum(dZ.T, axis=1)                       # (84,)
        
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA
    
    # Stochastic Diagonal Levenberg-Marquaedt
    def SDLM(self, d2Z, mu, lr_global):
        d2A = np.matmul(d2Z, np.power(self.weight.T,2))
        d2W = np.matmul(np.power(self.input_array.T,2), d2Z)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A
    
# not even slightly work
class RBFLayer_trainable_weight(object):
    def __init__(self, weight_shape, init_weight=None, init_mode='Gaussian_dist'): 
        self.weight_shape = weight_shape # =(10, 84)
        self.v_w = np.zeros(weight_shape)

        if init_weight.shape == (10,84):
            self.weight = init_weight
        else:
            self.weight, _ = initialize(weight_shape, init_mode)
        
    def foward_prop(self, input_array, label, mode): 
        """
        input_array = (n_m, 84)
        label = (n_m, )
        """
        
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label,:]  #(n_m, 84) labeled version of weight
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)  #(n_m, )
            return np.sum(np.squeeze(loss))
        
        if mode == 'test':
            subtract_weight = (input_array[:,np.newaxis,:] - np.array([self.weight]*input_array.shape[0])) # (n_m,10,84)
            rbf_class = np.sum(np.power(subtract_weight,2), axis=2) # (n_m, 10)
            class_pred = np.argmin(rbf_class, axis=1) # (n_m,)
            error01 = np.sum(label != class_pred)
            return error01, class_pred

    def back_prop(self, label, lr, momentum, weight_decay):
        #n_m = label.shape[0]
        
        #d_output = np.zeros((n_m, n_class))
        #d_output[range(n_m), label] = 1    # (n_m, 10)  one-hot version of gradient w.r.t. output
        
        dy_predict = -self.weight_label + self.input_array    #(n_m, 84)
        
        dW_target  = -dy_predict                              #(n_m, 84)
        
        dW = np.zeros(self.weight_shape) # (10,84)
        
        for i in range(len(label)):  
            dW[label[i],:] += dW_target[i,:]
            
        self.v_w = momentum*self.v_w - weight_decay*lr*self.weight - lr*dW
        self.weight += self.v_w

        return dy_predict

bitmap = rbf_init_weight()

class RBFLayer(object):
    def __init__(self, weight):        
        self.weight = weight  # (10, 84)
        
    def foward_prop(self, input_array, label, mode): 
        """
        input_array = (n_m, 84)
        label = (n_m, )
        """
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label,:]  #(n_m, 84) labeled version of weight
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)  #(n_m, )
            return np.sum(np.squeeze(loss))
        if mode == 'test':
            # (n_m,1,84) - n_m*[(10,84)] = (n_m,10,84)
            subtract_weight = (input_array[:,np.newaxis,:] - np.array([self.weight]*input_array.shape[0])) # (n_m,10,84)
            rbf_class = np.sum(np.power(subtract_weight,2), axis=2) # (n_m, 10)
            class_pred = np.argmin(rbf_class, axis=1) # (n_m,)
            error01 = np.sum(label != class_pred)
            return error01, class_pred
        
    def back_prop(self):
        dy_predict = -self.weight_label + self.input_array    #(n_m, 84)
        return dy_predict
    
    def SDLM(self):
        # d2y_predict
        return np.ones(self.input_array.shape)
    
    
class LeNet5(object):
    def __init__(self):
        kernel_shape = {"C1": (5,5,1,6),
                        "C3": (5,5,6,16),    ### C3 has designated combinations
                        "C5": (5,5,16,120),  ### It's actually a FC layer
                        "F6": (120,84),
                        "OUTPUT": (84,10)}
        
        hparameters_convlayer = {"stride": 1, "pad": 0}
        hparameters_pooling   = {"stride": 2, "f": 2}        
        
        self.C1 = ConvLayer(kernel_shape["C1"], hparameters_convlayer)
        self.a1 = Activation("LeNet5_squash")
        self.S2 = PoolingLayer(hparameters_pooling, "average")
        
        self.C3 = ConvLayer_maps(kernel_shape["C3"], hparameters_convlayer, C3_mapping)
        self.a2 = Activation("LeNet5_squash")
        self.S4 = PoolingLayer(hparameters_pooling, "average")
        
        self.C5 = ConvLayer(kernel_shape["C5"], hparameters_convlayer)
        self.a3 = Activation("LeNet5_squash")

        self.F6 = FCLayer(kernel_shape["F6"])
        self.a4 = Activation("LeNet5_squash")
        
        #self.Output = RBFLayer(kernel_shape["OUTPUT"], bitmap)
        self.Output = RBFLayer(bitmap)
        
    def Forward_Propagation(self, input_image, input_label, mode): 
        self.label = input_label
        self.C1_FP = self.C1.foward_prop(input_image)
        self.a1_FP = self.a1.foward_prop(self.C1_FP)
        self.S2_FP = self.S2.foward_prop(self.a1_FP)

        self.C3_FP = self.C3.foward_prop(self.S2_FP)
        self.a2_FP = self.a2.foward_prop(self.C3_FP)
        self.S4_FP = self.S4.foward_prop(self.a2_FP)

        self.C5_FP = self.C5.foward_prop(self.S4_FP)
        self.a3_FP = self.a3.foward_prop(self.C5_FP)

        self.flatten = self.a3_FP[:,0,0,:]
        self.F6_FP = self.F6.foward_prop(self.flatten)
        self.a4_FP = self.a4.foward_prop(self.F6_FP)  
        
        # output sum of the loss over mini-batch when mode = 'train'
        # output class when mode = 'test'
        out  = self.Output.foward_prop(self.a4_FP, input_label, mode) 

        return out 
        
    def Back_Propagation(self, momentum, weight_decay):
        dy_pred = self.Output.back_prop()
        
        dy_pred = self.a4.back_prop(dy_pred)
        F6_BP = self.F6.back_prop(dy_pred, momentum, weight_decay)
        reverse_flatten = F6_BP[:,np.newaxis,np.newaxis,:]
        
        reverse_flatten = self.a3.back_prop(reverse_flatten) 
        C5_BP = self.C5.back_prop(reverse_flatten, momentum, weight_decay)
        
        S4_BP = self.S4.back_prop(C5_BP)
        S4_BP = self.a2.back_prop(S4_BP)
        C3_BP = self.C3.back_prop(S4_BP, momentum, weight_decay) 
        
        S2_BP = self.S2.back_prop(C3_BP)
        S2_BP = self.a1.back_prop(S2_BP)  
        C1_BP = self.C1.back_prop(S2_BP, momentum, weight_decay)
        
    # Stochastic Diagonal Levenberg-Marquaedt method for determining the learning rate 
    def SDLM(self, mu, lr_global):
        d2y_pred = self.Output.SDLM()
        d2y_pred = self.a4.SDLM(d2y_pred)
        
        F6_SDLM = self.F6.SDLM(d2y_pred, mu, lr_global)
        reverse_flatten = F6_SDLM[:,np.newaxis,np.newaxis,:]
        
        reverse_flatten = self.a3.SDLM(reverse_flatten) 
        C5_SDLM = self.C5.SDLM(reverse_flatten, mu, lr_global)
        
        S4_SDLM = self.S4.SDLM(C5_SDLM)
        S4_SDLM = self.a2.SDLM(S4_SDLM)
        C3_SDLM = self.C3.SDLM(S4_SDLM, mu, lr_global)
        
        S2_SDLM = self.S2.SDLM(C3_SDLM)
        S2_SDLM = self.a1.SDLM(S2_SDLM)  
        C1_SDLM = self.C1.SDLM(S2_SDLM, mu, lr_global)
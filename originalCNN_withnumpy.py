from collections import OrderedDict
import numpy as np

from utils import check_conv_validity, check_pool_validity

np.random.seed(123)

def softmax(z):
    # Numerically stable softmax.
    z = z - np.max(z, axis=1, keepdims=True)
    _exp = np.exp(z)
    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum
    return sm

def zero_pad(x, pad):
   
    padded_x = None
    N, C, H, W = x.shape
   
    padded_x=np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    return padded_x


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        # if isinstance(kernel_size, int):
        #     kernel_size = (kernel_size, kernel_size)

        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
                batch_size, in_channel, _, _ = x.shape
        conv = self.convolution(x, self.w, self.b, self.stride, self.pad)
        self.output_shape = conv.shape
        return conv

    def convolution(self, x, w, b, stride=1, pad=0):
        #########################################################################################################
        # Convolution Operation.
        # 
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        # w: 4-D convolution filter
        # - Shape : (Out Channel, In Channel, Kernel Height, Kernel Width)
        # b: 1-D bias
        # - Shape : (Out Channel)
        # - default : None
        # stride : Stride size
        # - dtype : int
        # - default : 1
        # pad: pad value, how much to pad around
        # - dtype : int
        # - default : 0
        # 
        # [Output]
        # conv : convolution result
        # - Shape : (Batch size, Out Channel, Conv_Height, Conv_Width)
        # - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'Kernel Height', 'Kernel Width'
        #########################################################################################################
        
        
               
        a=int((H-HH)/stride+1)
        b=int((W-WW)/stride+1)
        
        conv=np.zeros((N,F,a,b))
       
        for first in range(N):
            for second in range(F):
                y_pos=out_y_pos=0
                while y_pos+HH<=H:
                    x_pos=out_x_pos=0
                    while x_pos+WW<=W:
                        conv[first,second,out_y_pos,out_x_pos]=(np.sum((w[second,:])*x[first,:,y_pos:y_pos+HH,x_pos:x_pos+WW]))#-b/a
                        x_pos +=stride
                        
                        out_x_pos +=1
                    y_pos +=stride
                    out_y_pos +=1

        return conv

    def backward(self, d_prev, reg_lambda):
        
        N, C, H, W = self.x.shape
        F, _, HH, WW = self.w.shape
        _, _, H_filter, W_filter = self.output_shape

        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)

        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.dx = np.zeros_like(self.x)
        temp=np.zeros_like(self.w)
        X_padding=np.pad(d_prev,( (0,0), (0,0),(self.pad,self.pad),(self.pad,self.pad)),'constant')
        dx=np.zeros(X_padding.shape)
        A2,B2,C2,D2=self.w.shape
        A,B,C3,D=d_prev.shape
        for i in range(N):
           for h in range(int((H-HH)/self.stride+1)):
               ih1=h*self.stride
               ih2=ih1+WW
               
               for w in range(int((W-WW)/self.stride+1)):
                   iw1=w*self.stride
                   iw2=iw1+WW
                   for f in range(A2): #WW
                       self.dx[i,:,ih1:ih2,iw1:iw2] +=self.w[f,:,:,:]*d_prev[i,f,h,w]
                       self.dw[f,:,:,:] +=self.x[i,:,ih1:ih2,iw1:iw2]*d_prev[i,f,h,w]
        self.dw +=reg_lambda*self.w
       
        self.dx=self.dx[:,:,self.pad:H-self.pad,self.pad:W-self.pad]
        
                        




        
        
        for j in range(B):
                self.db[j]=np.sum(d_prev[:,j])
        
        return self.dx

    def update(self, learning_rate):
        # Update weights
        self.w -= self.dw * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Filter Size : ' + str(self.w.shape) + ' Stride : %d, Zero padding: %d' % (self.stride, self.pad)


class MaxPoolingLayer:
    def __init__(self, kernel_size, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):

        max_pool = None
        N, C, H, W = x.shape
        check_pool_validity(x, self.kernel_size, self.stride)
        
        self.x = x
        new_H=int((H-self.kernel_size)/self.stride+1)
        new_W=int((W-self.kernel_size)/self.stride+1)
        max_pool=np.zeros((N,C,new_H,new_W))

        for first in range(N):
            for second in range(C):
                for third in range(new_H):
                    for fourth in range(new_W):
                        max_pool[first,second,third,fourth]=np.max(x[first,second,self.stride*third:self.stride*third+self.kernel_size,self.stride*fourth:self.stride*fourth+self.kernel_size])

        self.output_shape = max_pool.shape
        return max_pool

    def backward(self, d_prev, reg_lambda):

        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        N, C, H, W = d_prev.shape
        dx = np.zeros_like(self.x)

        for first in range(N):
            for second in range(C):
                for third in range(H):
                    for fourth in range(W):
                        for a2 in range(self.kernel_size):
                            for a3 in range(self.kernel_size):
                                if self.x[first,second,self.stride*third+a2,self.stride*fourth+a3]==np.max(self.x[first,second,self.stride*third:self.stride*third+self.kernel_size,self.stride*fourth:self.stride*fourth+self.kernel_size]):
                                    dx[first,second,self.stride*third+a2,self.stride*fourth+a3] +=d_prev[first,second,third,fourth]
        return dx


    def summary(self):
        return 'Pooling Size : ' + str((self.kernel_size, self.kernel_size)) + ' Stride : %d' % (self.stride)


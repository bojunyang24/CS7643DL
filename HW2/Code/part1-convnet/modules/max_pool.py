import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, h, w = x.shape
        s = self.stride
        k = self.kernel_size
        H_out = (h - k)// s + 1
        W_out = (w - k)// s + 1
        out = np.zeros((N, C, H_out, W_out))

        for h_ in range(H_out):
            for w_ in range(W_out):
                h1 = h_ * s
                h2 = h1 + k
                w1 = w_ * s
                w2 = w1 + k
                out[:,:, h_, w_] = np.max(x[:, :, h1:h2, w1:w2], axis=(2,3))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        k = self.kernel_size
        s = self.stride
        dx = np.zeros(x.shape)
        N, C, h, w = x.shape

        for h_ in range(H_out):
            for w_ in range(W_out):
                h1 = h_ * s
                h2 = h1 + k
                w1 = w_ * s
                w2 = w1 + k
                x_ = x[:, :, h1:h2, w1:w2]
                x_max = np.max(x_, axis=(2,3))
                mask = x_ == x_max[:,:,None,None]
                dx[:, :, h1:h2, w1:w2] += mask * dout[:, :, h_, w_][:, :, None, None]

        self.dx = dx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

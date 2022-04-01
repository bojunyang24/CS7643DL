import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        k = self.kernel_size
        p = self.padding
        s = self.stride
        N, C, h, w = x.shape
        F = self.weight.shape[0]
        h_new = (h - k + (2*p))//s + 1
        w_new = (w - k + (2*p))//s + 1
        x_ = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), 'constant')
        out = np.zeros((N, self.out_channels, h_new, w_new))

        # for h_ in range(0, h_new, self.stride):
        #     for w_ in range(0, w_new, self.stride):
        #         x_mask = x_[:,:,h_:h_+k,w_:w+k]
        #         for c in range(F):
        #             out[:,c,h_,w_] = np.sum(x_mask[:,c,:,:] * self.weight[c], axis=(1,2,3)) + self.bias[c]
                # out[n, :, h_, w_] = np.sum(np.multiply(x[n, :, h_:h_+k, w_:w_+k], self.weight[n, :]), axis=(1,2)) - self.bias
        

        for n in range(N):
            for h_ in range(h_new):
                for w_ in range(w_new):
                    for c in range(F):
                        h1 = h_ * s
                        h2 = h1 + k
                        w1 = w_ * s
                        w2 = w1 + k
                        out[n, c, h_, w_] = np.sum(np.multiply(x_[n, :, h1:h2, w1:w2], self.weight[c])) + self.bias[c]
                    # out[n, :, h_, w_] = np.sum(np.multiply(x_[n, :, h_:h_+k, w_:w_+k], self.weight[, :]), axis=(1,2)) + self.bias
        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        p = self.padding
        s = self.stride
        k = self.kernel_size
        x_ = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), 'constant')
        N, C, h, w = x.shape
        F = self.weight.shape[0]
        h_new = (h - k + (2*p))//s + 1
        w_new = (w - k + (2*p))//s + 1
        dx = np.zeros(x.shape)
        dw = np.zeros(self.weight.shape)
        db = np.sum(dout, axis=((0,2,3)))
        dx_ = np.zeros(x_.shape)

        for h_ in range(h_new):
            for w_ in range(w_new):
                h1 = h_ * s
                h2 = h1 + k
                w1 = w_ * s
                w2 = w1 + k
                for c in range(F):
                    dw[c, :, :, :] += np.sum(np.multiply(x_[:, :, h1:h2, w1:w2], dout[:, c, h_, w_][:, None, None, None]), axis = 0)
                for n in range(N):
                    dx_[n, :, h1:h2, w1:w2] += np.sum(
                        np.multiply(
                            self.weight,
                            dout[n, :, h_, w_][:, None, None, None]
                        ),
                        axis=0
                    )
        dx = dx_[:, :, p:-p,p:-p]
        self.dx = dx
        self.db = db
        self.dw = dw

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
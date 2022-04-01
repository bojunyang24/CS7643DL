from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                # i think i need to make a velocity tracker for each layer/module. so like grad_tracker[m]['v'] and track it like that
                
                if 'ws' not in self.grad_tracker.keys():
                    self.grad_tracker['ws'] = {}
                
                if m not in self.grad_tracker['ws'].keys():
                    self.grad_tracker['ws'][m] = [0]
                v_prev = self.grad_tracker['ws'][m][-1]
                try:
                    v = self.momentum * v_prev - self.learning_rate * m.dw
                except Exception as e:
                    print('hi')
                self.grad_tracker['ws'][m].append(v)
                m.weight += v
                # if 'wv' not in self.grad_tracker.keys():
                #     self.grad_tracker['wv'] = [0]
                # v_prev = self.grad_tracker['wv'][-1]
                # try:
                #     v = self.momentum * v_prev - self.learning_rate * m.dw
                # except Exception as e:
                #     print('hi')


                # self.grad_tracker['wv'].append(v)
                # m.weight += v
                
                # if 'vs' not in self.grad_tracker.keys():
                #     self.grad_tracker['vs'] = [0]
                # v_prev = self.grad_tracker['vs'][-1]
                # v = self.momentum * v_prev - self.learning_rate * m.dw
                # self.grad_tracker['vs'].append(v)
                # m.weight += v
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                
                if 'bs' not in self.grad_tracker.keys():
                    self.grad_tracker['bs'] = {}
                
                if m not in self.grad_tracker['bs'].keys():
                    self.grad_tracker['bs'][m] = [0]
                v_prev = self.grad_tracker['bs'][m][-1]
                try:
                    v = self.momentum * v_prev - self.learning_rate * m.db
                except Exception as e:
                    print('hi')
                self.grad_tracker['bs'][m].append(v)
                m.bias += v
                
                # if 'bv' not in self.grad_tracker.keys():
                #     self.grad_tracker['bv'] = [0]
                # v_prev = self.grad_tracker['bv'][-1]
                # v = self.momentum * v_prev - self.learning_rate * m.db
                # self.grad_tracker['bv'].append(v)
                # m.bias += v
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
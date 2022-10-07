import torch.nn as nn
import numpy
class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):# or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        print(count_targets)
        input()
        start_range = 1
        end_range = 1#count_targets-2#26#21#17count_targets-4#21#count_targets-2-11
        if start_range==end_range:
           self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range)\
                        .astype('int').tolist()
        else:

           self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        print(self.bin_range)
        res_conn = numpy.array([13,24])
        res_conn = res_conn.astype('int').tolist()
        print(res_conn)
        self.bin_range = (list(set(self.bin_range) - set(res_conn)))
        print(self.bin_range)
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        print(self.num_of_params)
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                print(m)
                #input()
                index = index + 1
                if index in self.bin_range and isinstance(m, nn.Conv2d):
                    print('Binarizing')
                    #input()
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        
        #self.save_params()#move here to save the original weights
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        kbit_conn = numpy.array([]) #you put which conv layers to make 2-bit/4-bit...kbit_conn=0 means the 2nd conv layer as first is anyways full-precision
        #[11,12,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32]
        kbit_conn = kbit_conn.astype('int').tolist()

        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            
            if index in kbit_conn:
            #k-bit quantization
                x = self.target_modules[index].data
                xmax = x.abs().max()
                num_bits=8
                v0 = 1
                v1 = 2
                v2 = -0.5
                y = 2.**num_bits - 1.
                x = x.add(v0).div(v1)
                x = x.mul(y).round_()
                x = x.div(y)
                x = x.add(v2)
                x = x.mul(v1)
                self.target_modules[index].data = x.mul(m.expand(s))
                '''
                #NOTE: alpha=WTQ/QTQ
                #result: 10 layer, 4-bit quantization: exit 22.2%, accuracy 64.09%
                WTQ = x.mul(self.target_modules[index].data).sum()
                QTQ = x.mul(x).sum()#element-wise multiplication, then sum all elements
                alpha = WTQ.div(QTQ)
                self.target_modules[index].data = x.mul(alpha.expand(s))
                '''
            else:
            #Binarize
                self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))
            
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            #TODO: if parameters fixed
            #if self.target_modules[index].grad==None:
            #    continue	
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

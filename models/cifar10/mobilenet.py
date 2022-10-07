import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['mobilenetv2']
class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        
        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6) 

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)
        
        #add extension
        self.corrector = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            LinearBottleNeck(32, 16, 1, 1),
            self._make_stage(1, 16, 24, 2, 6),
            self._make_stage(1, 24, 32, 2, 6),
            self._make_stage(1, 32, 64, 2, 6),
            self._make_stage(1, 64, 96, 1, 6),
            self._make_stage(1, 96, 160, 1, 6)
        )
        self.extension = self._make_stage(2, 160, 160, 1, 6)
        
        self.conv3 = nn.Sequential(
            LinearBottleNeck(160, 320, 1, 6),
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv4 = nn.Conv2d(1280, int(class_num/2), 1)

            
    def forward(self, x):
        inputx = x.clone().detach()
        #corrector
        y =self.corrector(inputx)
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        outputx = x.clone().detach()
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # return a list of probabilities
        output = []
        output.append(x)
        
        # Add layers
        #print(outputx.size(),y.size())
        xnew = outputx + y #add hard class correction
        #xnew = torch.cat((outputx, y),1) #concat
        exit1 = self.extension(xnew)
        exit1 = self.conv3(exit1)
        exit1 = F.adaptive_avg_pool2d(exit1, 1)
        exit1 = self.conv4(exit1)
        exit1 = exit1.view(exit1.size(0), -1)
        
        output.append(exit1)
        return output
    
    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        
        return nn.Sequential(*layers)

def mobilenetv2():
    return MobileNetV2()

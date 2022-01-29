import torch.nn as nn
class FeatureExtractor(nn.Module):
    def __init__(self, n_input):
        super(FeatureExtractor, self).__init__()
        feature = nn.Sequential()
        feature.add_module('f_fc1',  nn.Linear(n_input, 100))
        feature.add_module('f_relu1', nn.ReLU(True))
        feature.add_module('f_drop1', nn.Dropout())
        feature.add_module('f_fc2', nn.Linear(100, 50))
        feature.add_module('f_relu2', nn.ReLU(True))
        feature.add_module('f_drop2', nn.Dropout())
        self.feature = feature

    def forward(self, x):
        return self.feature(x)

class Cox(nn.Module):
    def __init__(self, n_input):
        super(Cox, self).__init__()
        self.cox_regression = nn.Sequential()
        # self.cox_regression.add_module('c_drop1', nn.Dropout())
        self.cox_regression.add_module('c_bn1', nn.BatchNorm1d(n_input))
        self.cox_regression.add_module('c_f2', nn.Linear(n_input, 1))
    def forward(self, x):
        return self.cox_regression(x)

class CPH_ROI(nn.Module):
    def __init__(self, device, n_input):
        super(CPH_ROI, self).__init__()
        self.device = device
        self.main = Cox(n_input)
        self.auxiliary = Cox(n_input)

        b_a = self.main.cox_regression.c_f2.bias
        self.auxiliary.cox_regression.c_f2.bias = b_a

    def forward(self, tar_fature, att_feature, istesting=False):
        if istesting:
            main_output = self.main(tar_fature)
            return main_output
        main_output = self.main(tar_fature)
        attentioner_output = self.auxiliary(att_feature)
        return main_output, attentioner_output

class CPH_DL(nn.Module):
    def __init__(self, device, n_input):
        super(CPH_DL, self).__init__()
        self.device = device
        self.feature = FeatureExtractor(n_input)
        self.Cox = Cox(50)

    def forward(self, tar_fature, istesting=False):
        if istesting:
            tar_fature = self.feature(tar_fature)
            tar_fature = tar_fature.view(-1, 50)
            main_output = self.Cox(tar_fature)
            return main_output

        tar_fature = self.feature(tar_fature)
        tar_fature = tar_fature.view(-1, 50)
        output = self.Cox(tar_fature)
        return output

class CPH_DL_ROI(nn.Module):
    def __init__(self, device, n_input):
        super(CPH_DL_ROI, self).__init__()
        self.device = device
        self.feature = FeatureExtractor(n_input)
        self.main = Cox(50)
        self.attentioner = Cox(50)

        b_a = self.main.cox_regression.c_f2.bias
        self.attentioner.cox_regression.c_f2.bias = b_a

    def forward(self, tar_fature, att_feature, istesting=False):
        if istesting:
            tar_fature = self.feature(tar_fature)
            tar_fature = tar_fature.view(-1, 50)
            main_output = self.main(tar_fature)
            return main_output

        tar_fature = self.feature(tar_fature)
        tar_fature = tar_fature.view(-1, 50)
        att_feature = self.feature(att_feature)
        att_feature = att_feature.view(-1, 50)

        main_output = self.main(tar_fature)
        attentioner_output = self.attentioner(att_feature)
        return main_output, attentioner_output

from torch import nn

def conv_L(in_len, kernel, stride, padding=0):
    return int((in_len - kernel + 2 * padding) / stride + 1)


def transpose_conv_output_length(input_length, kernel_size, stride, padding,
                                 output_padding, dilation):
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    output_length = (input_length - 1) * stride - 2 * padding + effective_kernel_size + output_padding
    return output_length


def cal_cnn_outlen(modules, in_len, pos:int|None=None):
    conv_l = in_len
    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, nn.ZeroPad3d):
                conv_l = in_len + m.padding[(2-pos)*2] + m.padding[(2-pos)*2+1]
                in_len = conv_l
            if isinstance(m, nn.ZeroPad2d):
                conv_l = in_len + m.padding[(1-pos)*2] + m.padding[(1-pos)*2+1]
                in_len = conv_l
            if isinstance(m, nn.Conv1d):
                conv_l = conv_L(in_len, m.kernel_size[0], m.stride[0], m.padding[0])
                in_len = conv_l
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos])
                in_len = conv_l
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                conv_l = transpose_conv_output_length(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos],
                                                      m.output_padding[pos], m.dilation[pos])
                in_len = conv_l
            if isinstance(m, nn.AvgPool1d) or isinstance(m, nn.MaxPool1d):
                conv_l = conv_L(in_len, m.kernel_size, m.stride, m.padding)
                in_len = conv_l
            if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
                conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos])
                in_len = conv_l
    elif isinstance(modules, nn.ModuleList):
        for layer in modules:
            for m in layer:
                if isinstance(m, nn.ZeroPad3d):
                    conv_l = in_len + m.padding[(2 - pos) * 2] + m.padding[(2 - pos) * 2 + 1]
                    in_len = conv_l
                if isinstance(m, nn.ZeroPad2d):
                    conv_l = in_len + m.padding[(1 - pos) * 2] + m.padding[(1 - pos) * 2 + 1]
                    in_len = conv_l
                if isinstance(m, nn.Conv1d):
                    conv_l = conv_L(in_len, m.kernel_size[0], m.stride[0], m.padding[0])
                    in_len = conv_l
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos])
                    in_len = conv_l
                if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                    conv_l = transpose_conv_output_length(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos],
                                                          m.output_padding[pos], m.dilation[pos])
                    in_len = conv_l
                if isinstance(m, nn.AvgPool1d) or isinstance(m, nn.MaxPool1d):
                    conv_l = conv_L(in_len, m.kernel_size, m.stride, m.padding)
                    in_len = conv_l
                if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
                    conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos])
                    in_len = conv_l

    elif isinstance(modules, nn.Module):
        for m in modules.modules():
            if isinstance(m, nn.Sequential):
                for m_1 in m:
                    if isinstance(m_1, nn.Conv1d):
                        conv_l = conv_L(in_len, m_1.kernel_size[0], m_1.stride[0], m_1.padding[0])
                        in_len = conv_l
                    if isinstance(m_1, nn.Conv2d) or isinstance(m_1, nn.Conv3d):
                        conv_l = conv_L(in_len, m_1.kernel_size[pos], m_1.stride[pos], m_1.padding[pos])
                        in_len = conv_l
                    if isinstance(m_1, nn.ConvTranspose2d) or isinstance(m_1, nn.ConvTranspose3d):
                        conv_l = transpose_conv_output_length(in_len, m_1.kernel_size[pos], m_1.stride[pos], m_1.padding[pos],
                                                              m_1.output_padding[pos], m_1.dilation[pos])
                        in_len = conv_l
                    if isinstance(m_1, nn.AvgPool1d) or isinstance(m_1, nn.MaxPool1d):
                        conv_l = conv_L(in_len, m_1.kernel_size, m_1.stride, m_1.padding)
                        in_len = conv_l
                    if isinstance(m_1, nn.AvgPool2d) or isinstance(m_1, nn.MaxPool2d):
                        conv_l = conv_L(in_len, m_1.kernel_size[pos], m_1.stride[pos], m_1.padding[pos])
                        in_len = conv_l
            if isinstance(m, nn.Conv1d):
                conv_l = conv_L(in_len, m.kernel_size[0], m.stride[0], m.padding[0])
                in_len = conv_l
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos])
                in_len = conv_l
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                conv_l = transpose_conv_output_length(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos],
                                                      m.output_padding[pos], m.dilation[pos])
                in_len = conv_l
            if isinstance(m, nn.AvgPool1d) or isinstance(m, nn.MaxPool1d):
                conv_l = conv_L(in_len, m.kernel_size, m.stride, m.padding)
                in_len = conv_l
            if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
                conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[pos], m.padding[pos])
                in_len = conv_l

    else:
        if isinstance(modules, nn.Conv1d) or isinstance(modules, nn.Conv2d) or isinstance(modules, nn.Conv3d):
            conv_l = conv_L(in_len, modules.kernel_size[pos], modules.stride[pos], modules.padding[pos])
        if isinstance(modules, nn.ConvTranspose2d) or isinstance(modules, nn.ConvTranspose3d):
            conv_l = transpose_conv_output_length(in_len, modules.kernel_size[pos], modules.stride[pos], modules.padding[pos],
                                                  modules.output_padding[pos], modules.dilation[pos])
        if isinstance(modules, nn.AvgPool1d) or isinstance(modules, nn.MaxPool1d):
            conv_l = conv_L(in_len, modules.kernel_size, modules.stride, modules.padding)
        if isinstance(modules, nn.AvgPool2d) or isinstance(modules, nn.MaxPool2d):
            conv_l = conv_L(in_len, modules.kernel_size[pos], modules.stride[pos], modules.padding[pos])
    return conv_l
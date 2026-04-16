from models.conv5 import Conv5, Conv5_small, LeNet5, SimpleMLP, MiniCNN, ResNet8, ResNet20, MobileNetV2_Tiny, GoogLeNet, AlexNet


def get_model(model_name, loss_type='fedavg'):
    if model_name == "conv5":
        return Conv5(loss_type=loss_type)
    elif model_name == "conv5small":
        return Conv5_small(loss_type=loss_type)
    elif model_name == "lenet5":
        return LeNet5(loss_type=loss_type)
    elif model_name == "simplemlp":
        return SimpleMLP(loss_type=loss_type)
    elif model_name == "minicnn":
        return MiniCNN(loss_type=loss_type)
    elif model_name == "resnet8":
        return ResNet8(loss_type=loss_type)
    elif model_name == "resnet20":
        return ResNet20(loss_type=loss_type)
    elif model_name == "mobilenetv2":
        return MobileNetV2_Tiny(loss_type=loss_type)
    elif model_name == "googlenet":
        return GoogLeNet(loss_type=loss_type)
    elif model_name == "alexnet":
        return AlexNet(loss_type=loss_type)
    else:
        raise NotImplementedError(f'[!] ERROR: Model {model_name} not implemented yet')

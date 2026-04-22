from models.conv5 import Conv5, Conv5_small, LeNet5, LeNet5_slim, SimpleMLP, MiniCNN, ResNet8, ResNet20, MobileNetV2_Tiny, GoogLeNet, AlexNet


def get_model(model_name, loss_type='fedavg'):
    if model_name == "conv5":
        return Conv5(loss_type=loss_type)
    elif model_name == "conv5small":
        return Conv5_small(loss_type=loss_type)
    elif model_name == "lenet5":
        return LeNet5(loss_type=loss_type)
    elif model_name == "lenet5_slim":
        # Expects 16x16 downsampled input (use with load_data(resize=16))
        return LeNet5_slim(loss_type=loss_type, downsampled=True)
    elif model_name == "lenet5_slim_32":
        # Same slim architecture but expects standard 32x32 input
        return LeNet5_slim(loss_type=loss_type, downsampled=False)
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

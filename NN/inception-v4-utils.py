def choose_nets(nets_name, num_classes=100):
    nets_name = nets_name.lower()
    if nets_name == 'inceptionv4':
        from models.InceptionV4 import inceptionv4
        return inceptionv4(num_classes)
    raise NotImplementedError

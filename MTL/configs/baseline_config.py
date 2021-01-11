class Params():
    def __init__(self):
        self.is_param = True

# python main.py --config=mini_5_resnet_baseline --gpu=0 --num_workers=16
def mini_5_resnet_baseline():
    param = Params()
    param.shot = 5
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "miniImagenet"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/mini/softmax/resnet10/model_best.pth.tar"
    param.meta_label = "resmini"
    param.dacc = False
    return param

def mini_1_resnet_baseline():
    param = Params()
    param.shot = 1
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "miniImagenet"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/mini/softmax/resnet10/model_best.pth.tar"
    param.meta_label = "resmini"
    param.dacc = False
    return param

def tiered_5_resnet_baseline():
    param = Params()
    param.shot = 5
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "tiered"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/tiered/ResNet10/model_best.pth.tar"
    param.meta_label = "restiered"
    param.dacc = False
    return param

def tiered_1_resnet_baseline():
    param = Params()
    param.shot = 1
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "tiered"
    param.method = "simpleshot"
    param.model = "ResNet10"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/tiered/ResNet10/model_best.pth.tar"
    param.meta_label = "restiered"
    param.dacc = False
    return param

def mini_5_wrn_baseline():
    param = Params()
    param.shot = 5
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "miniImagenet"
    param.method = "simpleshotwide"
    param.model = "wideres"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/mini/softmax/wideres/model_best.pth.tar"
    param.meta_label = "wrnmini"
    param.dacc = False
    return param

def mini_1_wrn_baseline():
    param = Params()
    param.shot = 1
    param.test_iter = 600
    param.test = True
    param.debug = False
    param.dataset = "miniImagenet"
    param.method = "simpleshotwide"
    param.model = "wideres"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/mini/softmax/wideres/model_best.pth.tar"
    param.meta_label = "wrnmini"
    param.dacc = False
    return param

def tiered_5_wrn_baseline():
    param = Params()
    param.shot = 5
    param.test_iter = 2000
    param.test = True
    param.debug = False
    param.dataset = "tiered"
    param.method = "simpleshotwide"
    param.model = "wideres"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/tiered/wideres/model_best.pth.tar"
    param.meta_label = "wrntiered"
    param.dacc = False
    return param

def tiered_1_wrn_baseline():
    param = Params()
    param.shot = 1
    param.test_iter = 600
    param.test = True
    param.debug = False
    param.dataset = "tiered"
    param.method = "simpleshotwide"
    param.model = "wideres"
    param.deconfound = False
    param.init_weights = "/nas/people/lukasz_bala/reproducibility/ifsl/downloads/models/results/tiered/wideres/model_best.pth.tar"
    param.meta_label = "wrntiered"
    param.dacc = False
    return param
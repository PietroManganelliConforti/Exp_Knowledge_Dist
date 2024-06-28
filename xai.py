from pytorch_grad_cam import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def return_cam_from_model(model, target_layer, batch, targets , cam_name = "gradcam"):

    model.eval() # da rimettere in .train() se nel training loop

    cam = None

    if cam_name == "gradcam":
        cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

    elif cam_name == "scorecam":
        cam = ScoreCAM(model=model, target_layers=target_layer, use_cuda=True) # CONTROLLARE CHE NON VA

    elif cam_name == "fullcam":
        cam = FullGrad(model=model, target_layers=target_layer, use_cuda=True)

    elif cam_name == 'gradcamplspls':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layer, use_cuda=True)
    
    elif cam_name == 'xgradcam':
        cam = XGradCAM(model=model, target_layers=target_layer, use_cuda=True)
        
    elif cam_name == 'eigencam':
        cam = EigenCAM(model=model, target_layers=target_layer, use_cuda=True)
     
    elif cam_name == 'eigengradcam':
        cam = EigenGradCAM(model=model, target_layers=target_layer, use_cuda=True)
        
    elif cam_name == 'layercam':
        cam = LayerCAM(model=model, target_layers=target_layer, use_cuda=True)

    elif cam_name == 'fullgrad':
        cam = FullGrad(model=model, target_layers=target_layer, use_cuda=True)
     
    elif cam_name == 'hirescam':
        cam = HiResCAM(model=model, target_layers=target_layer, use_cuda=True)
        
    elif cam_name == 'gradcamelementwise':
        cam = GradCAMElementWise(model=model, target_layers=target_layer, use_cuda=True)

    else:
        raise Exception("Cam name not recognized")
    
    targets = [ClassifierOutputTarget(i) for i in targets]

    output_cam = cam(input_tensor=batch, targets=targets)

    model.train()

    return output_cam
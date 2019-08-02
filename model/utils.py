try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class LayerActivations():
    features=[]
    def __init__(self,model):
        self.features = []
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.extend(output.view(output.size(0),-1).cpu().data)

    def remove(self):
        self.hook.remove()

from ..VisualCodingBlocks.Text2Mask import Text2Mask

class Mask:
    def __init__(self,depth_scape) :
        self.depthScape = depth_scape
    def get_mask(self,text):
        self.text = text
        self.mask=Text2Mask()
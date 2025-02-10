class Mask:
    def __init__(self,depth_scape,text,mask) :
        self.depthScape = depth_scape
        self.mask = mask
        self.text=text
    def show_mask(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.mask)
        plt.show()
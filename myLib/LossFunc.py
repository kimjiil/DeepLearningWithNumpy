if __name__ == "__main__":
    from Module import myModule, Parameter
else:
    from .Module import myModule, Parameter
    
class testloss(myModule):
    def __init__(self):
        super(testloss, self).__init__()


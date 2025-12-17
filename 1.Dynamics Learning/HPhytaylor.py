import RTaylorNN as tnn

class HPhytaylor:
    def __init__(self, RtaylorNN, xs,t,meanxs,Project):
        self.datainput = xs
        self.dataoutput = t
        self.meanxs= meanxs
        self.Project= Project;
        self.tnnlist =[]

    def add_rtaylor(self,tnn):
        self.tnnlist.append(tnn)

    def generateTaylor(self,params):
        x, y, ly, weights, biases = tnn.create_DeepTaylor_net(params)
    return

    def MEParititions(self,Bound,threshold):

    def Merging(self,threshold):

    def Learn(self,Partitions):

    def Predict(self,input,Partitions):

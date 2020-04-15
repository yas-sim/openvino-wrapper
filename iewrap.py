import os

import cv2
from openvino.inference_engine import IENetwork, IECore

class ieWrapper:

    def __init__(self, modelFile=None, device='CPU', numRequest=4):
        self.ie          = IECore()
        self.inferSlot   = 0
        self.callbackFunc= None
        self.inferenceID = 0
        if modelFile == None:
            self.inputName   = None
            self.outputName  = None
            self.N           = None
            self.C           = None
            self.H           = None
            self.W           = None
            self.execNet     = None   # exec_net
            self.numRequests = 0
        else:
            fname, ext = os.path.splitext(modelFile)
            self.readModel(fname+'.xml', fname+'.bin', device, numRequest)

    def readModel(self, xmlFile, binFile, device='CPU', numRequest=4):
        net = self.ie.read_network(xmlFile, binFile)
        self.inputName  = list(net.inputs.keys())[0]
        self.outputName = list(net.outputs.keys())[0]
        self.N, self.C, self.H, self.W = net.inputs[self.inputName].shape
        self.execNet = self.ie.load_network(network=net, device_name=device, num_requests=numRequest)
        self.numRequests = numRequest
        #del net

    def callback(self, status, data):
        id, req = data
        output = req.outputs[self.outputName]
        if self.callbackFunc!=None:
            self.callbackFunc(id, output)

    def imagePreprocess(self, img):
        img = cv2.resize(img, (self.W, self.H))
        img = img.transpose((2, 0, 1))
        img = img.reshape((self.N, self.C, self.H, self.W))
        return img

    def asyncInfer(self, img):
        status = None
        while status!=0 and status!=-11:
            req = self.execNet.requests[self.inferSlot]
            self.inferSlot = (self.inferSlot+1) % self.numRequests
            status = req.wait(-1)
        infID = self.inferenceID
        req.set_completion_callback(self.callback, (infID, req))
        req.async_infer(inputs={ self.inputName: self.imagePreprocess(img) })
        self.inferenceID+=1
        return infID
    
    def setCallback(self, callback):
        self.callbackFunc = callback
    
    def waitForAllCompletion(self):
        while True:
            numIdle=0
            for i in range(self.numRequests):
                status = self.execNet.requests[i].wait(-1)
                if status == 0 or status==-11:
                    numIdle+=1
            if numIdle == self.numRequests:
                return

    def blockInfer(self, img):
        res = self.execNet.infer(inputs={ self.inputName: self.imagePreprocess(img) })
        return res[self.outputName]

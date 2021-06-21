from random import randint
from math import exp, isnan
import os
class NETWORK(object):
    def __init__(self, inputs, HiddenNumber, OutputWanted, LearningRate, log=True, AcceptError=0.25, sigmoide=1):
        self.Log = log
        self.inputs = inputs
        self.InputsNumber = len(self.inputs)
        self.hiddens = []
        self.output = 0.00
        self.OutputWanted = OutputWanted
        self.LearningRate = LearningRate
        self.HiddenNumber = HiddenNumber
        self.OutputNumber = 1
        self.NetworkResult = 0.00
        self.AcceptError = AcceptError
        self.sigmoide = sigmoide
        self.OutputNeuralError = 100000
        self.RetroNeuralOutput = 0.00
        self.w1 = []
        self.w2 = []
        self.OutputW1 = []
        self.OutputW2 = []
        self.RetroW2Output = []
        self.RetroW1Output = []
        self.RetroHiddensOutput = []
        del inputs, HiddenNumber, OutputWanted, LearningRate, log, AcceptError, sigmoide
        self.InitForFirstTimeNetwork()
        self.ShowSettings()
    def log(self, txt):
        if self.Log: print(txt)
        else: pass
    def InitForFirstTimeNetwork(self):
        #On initialise les poids pour la première fois
        if len(self.w1): self.w1.clear()
        if len(self.w2): self.w2.clear()
        if len(self.hiddens): self.hiddens.clear()
        if len(self.OutputW1): self.OutputW1.clear()
        if len(self.OutputW2): self.OutputW2.clear()
        if len(self.RetroHiddensOutput): self.RetroHiddensOutput.clear()
        if len(self.RetroW2Output): self.RetroW2Output.clear()
        if len(self.RetroW1Output): self.RetroW1Output.clear()
        for i in range(0, self.InputsNumber * self.HiddenNumber): self.w1.append(self.w())
        for i in range(0, self.HiddenNumber * self.OutputNumber): self.w2.append(self.w())
        for i in range(0, self.HiddenNumber): self.hiddens.append(0.00)
        for i in range(0, len(self.w1)): self.OutputW1.append(0.00)
        for i in range(0, len(self.w2)): self.OutputW2.append(0.00)
        for i in range(0, len(self.w2)): self.RetroW2Output.append(0.00)
        for i in range(0, len(self.hiddens)): self.RetroHiddensOutput.append(0.00)
        for i in range(0, len(self.w1)): self.RetroW1Output.append(0.00)
        for i in range(0, len(self.inputs)): self.inputs[i] = self.sigmoid(self.inputs[i])
        self.output = 0.00 #Déjà fait dans __init__
    def w(self): return randint(-1, 1) + (randint(0, 100) / 100)
    def ShowSettings(self):
        self.log("Entrées du réseau >>> " + str(self.inputs) + " (" + str(self.InputsNumber) + " neurones en entrée)")
        self.log("Poids entre la première et la seconde couche >>> " + str(self.w1) + " (" + str(len(self.w1)) + " poids)")
        self.log("Neurones de la seconde couche >>> " + str(self.hiddens) + " (" + str(self.HiddenNumber) + " neurones)")
        self.log("Poids entre la seconde et la dernière couche >>> " + str(self.w2) + " (" + str(len(self.w2)) + " poids)")
        self.log("Neurone de sortie >>> " + str(self.output))
        self.log("Sortie souhaitée >>> " + str(self.OutputWanted))
        self.log("Sortie du réseau (qui est la valeur du dernier neurone) >>> " + str(self.NetworkResult))
    def Feedforward(self):
        # os.system('clear')
        self.FeedforwardInputsNeurals()
        self.FeedforwardFirstSynapses()
        self.FeedforwardHiddensNeurals()
        self.FeedforwardSecondesSynapses()
        self.FeedforwardOutputNeural()
        self.SetNetworkResult()
        self.log("Après ce Feedforward, le résultat du réseau est : " + str(self.NetworkResult))
    def FeedforwardInputsNeurals(self): self.log("Les neurones en entrée envoient leurs valeurs aux " + str(len(self.w1)) + " synapses ...") #Ils n'envoient que leurs valeurs
    def FeedforwardFirstSynapses(self):
        self.log("Les premières synapses calculent leurs valeurs ...")
        input_signal = self.inputs #C'est le courrier envoyé par les neurones en entrée x)
        current_weights = self.w1
        output_signal = []
        #i % self.InputsNumber permet de savoir à quel neurone d'entrée correspond quelle synapse
        for i in range(0, len(current_weights)): output_signal.append(current_weights[i] * input_signal[i % self.InputsNumber])
        self.OutputW1 = output_signal
        self.log("Elles envoient : " + str(self.OutputW1))
    def FeedforwardHiddensNeurals(self):
        self.log("Les neurones de la seconde couche calculent leurs valeurs ...")
        input_signal = self.OutputW1
        hiddens_output = self.hiddens
        for i in range(0, len(input_signal)):
            hiddens_output[i % self.HiddenNumber] += input_signal[i]
        for i in range(0, self.HiddenNumber):
            if hiddens_output[i] < 0: hiddens_output[i] = 0
            else: pass
        self.hiddens = hiddens_output
        self.log("Ils envoient aux synapses suivantes : " + str(self.hiddens))
    def FeedforwardSecondesSynapses(self):
        self.log("Les secondes synapses calculent leurs valeurs ...")
        input_signal = self.hiddens
        current_weights = self.w2
        output_signal = []
        for i in range(0, len(current_weights)): output_signal.append(current_weights[i] * input_signal[i % self.HiddenNumber])
        self.OutputW2 = output_signal
        self.log("Elles envoient au dernier neurone : " + str(self.OutputW2))
    def FeedforwardOutputNeural(self):
        self.log("Le neurone de sortie calcule sa valeur ...")
        input_signal = self.OutputW2
        output_signal = 0.00
        for i in range(0, len(input_signal)): output_signal += input_signal[i]
        self.output = output_signal
        self.log("Sa valeur est de : " + str(self.output))
    def SetNetworkResult(self): self.NetworkResult = self.output
    def Backpropagation(self):
        self.RetroOutputNeural()
        self.RetroSecondesSynapses()
        self.RetroHiddensNeurals()
        self.RetroFirstSynapses()
        self.RetroInputsNeural()
    def RetroOutputNeural(self):
        self.log("Le neurone de sortie calcule son erreur et la donnée à transmettre ...")
        self.OutputNeuralError = self.OutputWanted - self.NetworkResult
        self.log("L'erreur est de : " + str(self.OutputNeuralError))
        self.RetroNeuralOutput = self.OutputNeuralError * self.LearningRate
        self.log("Le neurone de sortie envoie aux dernières synapses : " + str(self.RetroNeuralOutput))
    def RetroSecondesSynapses(self):
        self.log("Les dernières synapses se corrigent ...")
        input_signal = self.RetroNeuralOutput
        current_weights = self.w2
        output_signal = []
        for i in range(0, len(current_weights)): output_signal.append(current_weights[i] * input_signal)
        self.RetroW2Output = output_signal
        self.log("Les synapses envoient aux neurones de la seconde couche : " + str(self.RetroW2Output))
        self.log("Elles modifient leurs poids ...")
        for i in range(0, len(current_weights)): current_weights[i] += (self.OutputW2[i] * input_signal)
        self.w2 = current_weights
        self.log("Les nouveaux poids des synapses sur la seconde couche sont : " + str(self.w2))
    def RetroHiddensNeurals(self):
        self.log("Les neurones sur la seconde couche calculent leurs nouvelles valeurs ...")
        input_signal = self.RetroW2Output
        output_signal = self.RetroHiddensOutput
        for i in range(0, len(self.RetroW2Output)): output_signal[i % self.HiddenNumber] += input_signal[i]
        for i in range(0, len(self.hiddens)):
            if abs(self.hiddens[i]) > 100000: self.InitForFirstTimeNetwork()
        self.RetroHiddensOutput = output_signal
        self.log("Les neurones sur la seconde couche envoit vers les premières synapses : " + str(self.RetroHiddensOutput))
    def RetroFirstSynapses(self):
        self.log("Les premières synapses se corrigent ...")
        input_signal = self.RetroHiddensOutput
        current_weights = self.w1
        output_signal = []
        for i in range(0, len(current_weights)): output_signal.append(current_weights[i] * input_signal[i % self.HiddenNumber])
        self.RetroW1Output = output_signal
        self.log("Les synapses envoient aux neurones d'entrée : " + str(self.RetroW1Output))
        self.log("Elles modifient leurs poids ...")
        for i in range(0, len(current_weights)): current_weights[i] += (self.OutputW1[i] * input_signal[i % self.HiddenNumber])
        self.w1 = current_weights
        self.log("Les nouveaux poids des synapses sur la première couche sont : " + str(self.w1))
    def RetroInputsNeural(self): pass #Ils n'ont rien à faire
    def sigmoid(self, x):
        if self.sigmoide:
            return 1/(1 + exp(-x))
        else: return x
    def Train(self):
        while abs(self.OutputNeuralError) > self.AcceptError:
            self.Feedforward()
            self.Backpropagation()
            if isnan(self.NetworkResult):
                '''self.InitForFirstTimeNetwork()
                self.OutputNeuralError = 12356'''
                exit()
            else:
                print("Erreur actuelle : " + str(self.OutputNeuralError))
    def Play(self, datas):
        if len(datas) != self.InputsNumber:
            print("Le nombre de données doit être égal à : " + str(self.InputsNumber) + " !")
            exit()
        else:
            self.inputs = datas
            self.Feedforward()
            return self.sigmoid(self.NetworkResult)
'''N = NETWORK([1, 2, 3, 4, 56, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 2, 0.47, 0.04, 1, AcceptError=0.01)
N.Train()
print(N.Play([1, 2, 5, 6, 4, 1, 2, 5, 6, 4, 1, 2, 5, 6, 4, 1, 2, 5, 6]))'''
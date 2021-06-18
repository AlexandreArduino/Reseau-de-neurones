from random import randint

'''
BAALBAKY Alexandre
Ce projet n'a pas de license mais je vous demande juste de mentionner l'auteur principal du code avant de le modifier :)
'''

class NETWORK(object):
    def __init__(self, InputsNumber, HiddensNumber, OutputWeWant, LearningRate):
        self.inputs = []
        self.hiddens = []
        self.output = 0.00#On attend une seule sortie
        self.w1 = []
        self.w2 = []
        self.ow1 = [] #Résultat des synapses
        self.ow2 = []
        self.rhiddens = []
        self.rinputs = []
        self.InputsNumber = InputsNumber
        self.HiddensNumber = HiddensNumber
        self.OutputWeWant = OutputWeWant
        self.retro = 0 #Signal à relayer par le dernier neurone pendant l'apprentissage (backpropagation)
        self.rw2 = []
        self.rw1 = []
        del InputsNumber, HiddensNumber
        self.InitInputsAndHiddensForFirstTime()
        self.InitWeightsRandomlyForFirstTime()
        self.LearningRate = LearningRate
    def InitInputsAndHiddensForFirstTime(self):
        print("Initialisation des valeurs des neurones à 0.00 ...")
        for i in range(0, self.InputsNumber): self.inputs.append(0.00)
        for i in range(0, self.HiddensNumber): self.hiddens.append(0.00)
        self.rinputs = self.inputs
        self.rhiddens = self.hiddens
        print("Neurones en entrée (neurones initiaux) >>> " + str(self.inputs))
        print("Neurones en seconde couche (neurones intermédiaires) >>> " + str(self.hiddens))
        print("Neurone en sortie >>> " + str(self.output))
        del i
    def InitWeightsRandomlyForFirstTime(self):
        print("Initialisation des poids pour la première fois de manière aléatoire ...")
        for i in range(0, self.InputsNumber*self.HiddensNumber): self.w1.append(self.w())
        for i in range(0, self.HiddensNumber): self.w2.append(self.w())
        self.ow1 = self.w1
        self.ow2 = self.w2 #C'est temporaire, c'est juste pour ne pas avoir 0.00
        self.rw1 = self.w1 #pareil
        self.rw2 = self.w2
        print("Poids entre la couche en entrée et la seconde couche (synapses) >>> " + str(self.w1))
        print("Poids entre la seconde couche et le neurone en sortie (synapses) >>> " + str(self.w2))
    def w(self): return randint(0, 1) + (randint(-100, 100)/100)
    def SetInputs(self, inputs):
        print("Mise à jour des neurones initiaux ...")
        if len(inputs) != self.InputsNumber:
            print("Erreur, le nombre de données n'est pas égal à celui du nombre de neurones initiaux!")
        else: self.inputs = inputs
        print("Neurones initiaux >>> " + str(self.inputs))
    def Feedforward(self):
        print("Propagation avant (Feedforward) ...")
        #On commence par calculer le résultat des synapses entre la première et la seconde couche
        self.CalculateSynapsesResultBetweenFirstAndSecondLayer()
        #À partir de là, la sortie des synapses est l'ensemble des résultats qu'on vient de déterminer
        #On lance donc le calcul des valeurs des neurones de la seconde couche
        self.AddSynapsesResultForHiddensNeural()
        self.DetermineOutputHiddenNeural()
        #On calcule la sortie des synapses suivantes
        self.CalculateSynapsesResultBetweenSecondeAndLastLayer()
        #On fait la somme de celle-ci pour le dernier neurone
        self.AddSynapsesResultForOutputNeural()
    def CalculateSynapsesResultBetweenFirstAndSecondLayer(self):
        print("\t- Calcul des résultats des synapses entre la première et la seconde couche ...")
        for synapse in range(0, len(self.w1)):
            self.ow1[synapse] = self.inputs[synapse % self.InputsNumber] * self.w1[synapse] #On rappelle que self.ow1 contient les résultats des synapses entre la première et la seconde couche
        print("\t\t- Résultat : " + str(self.ow1))
    def AddSynapsesResultForHiddensNeural(self):
        print("\t- Calcul des sommes des signaux reçus par chaque neurone (caché) de la seconde couche ...")
        for i in range(0, len(self.ow1)):
            self.hiddens[i % self.HiddensNumber] += self.ow1[i]
        print("\t\t- Résultat obtenu >>> " + str(self.hiddens))
    def DetermineOutputHiddenNeural(self):
        print("\t- Calcul du message de sortie des neurones de la seconde couche ... (avec le signe de la somme)")
        for i in range(0, self.HiddensNumber):
            if self.hiddens[i] < 0: self.hiddens[i] = 0
            else: pass
        print("\t\t- Sortie finale des neurones de la seconde couche >>> " + str(self.hiddens))
    def CalculateSynapsesResultBetweenSecondeAndLastLayer(self):
        print("\t- Calcul des résultats des synapses entre la seconde et la dernière couche ...")
        for synapse in range(0, len(self.w2)):
            self.ow2[synapse] = self.hiddens[synapse % self.HiddensNumber] * self.w2[synapse] #On rappelle que self.ow1 contient les résultats des synapses entre la première et la seconde couche
        print("\t\t- Résultat : " + str(self.ow2))
    def AddSynapsesResultForOutputNeural(self):
        print("\t- Calcul des sommes des signaux reçus pour le dernier neurone ...")
        for i in range(0, len(self.ow2)): self.output += self.ow2[i]
        print("\t\t- Résultat obtenu du réseau >>> " + str(self.output))
    def Backpropagation(self):
        print("Propagation arrière (Backpropagation) permettant l'apprentissage du réseau ...")
        print("\t- Calcul de l'erreur ...")
        self.error = self.OutputWeWant - self.output
        print("\t\t- Erreur obtenu >>> " + str(self.error))
        #On corrige l'erreur (pour plus de détails : https://www.youtube.com/watch?v=5vKWO3qnHyY à 19:40)
        self.retro = self.error * self.LearningRate
        print("\t\t- Résultat rétropropagation du dernier neurone >>> " + str(self.retro))
        print("\t- Calcul des nouvelles valeurs des synapses entre la seconde et la dernière couche du réseau ...")
        for i in range(0, len(self.w2)):
            self.rw2[i] = self.w2[i] * self.retro
            self.NewWeigth = self.ow2[i] * self.retro
            self.w2[i] += self.NewWeigth
        last_hiddens = self.hiddens
        print("\t\t- Résultat obtenu >>> " + str(self.w2))
        print("\t- Calcul de la nouvelles valeur des neurones de la seconde couche ...")
        for i in range(0, len(self.hiddens)):
            #Il n'y a qu'un neurone en sortie donc qu'une seule synapse par neurone à droite sur la seconde couche
            self.hiddens[i] = self.rw2[i] #Toujours car il n'y a qu'une seul neurone
            if not last_hiddens[i]: self.hiddens[i] = 0
            else: pass
        print("\t\t- Résultat obtenu >>> " + str(self.hiddens))
        #On finit pas réadapter les poids w1 des synapses entre la première et la seconde couche
        print("\t- Calcul des nouvelles valeurs des synapses entre la première et la seconde couche du réseau ...")
        for i in range(0, len(self.w1)):
            self.retro = self.hiddens[i % self.HiddensNumber]
            self.rw1[i] = self.w1[i] * self.retro
            self.NewWeigth = self.ow1[i] * self.retro
            print(self.w1[i])
            self.w1[i] += self.NewWeigth
            print(self.w1[i])
        print("\t\t- Résultat obtenu >>> " + str(self.w1))
    def Train(self):
        for i in range(100):
            self.Feedforward()
            input("Appuyez pour l'entrainement n°" + str(i + 1))
            self.Backpropagation()
            input()
N = NETWORK(len([1,5,6,4,7,8,9,8,7,6,5,4,1,2,3,6,5,4]), 2, 1, 0.04)
N.SetInputs([1,5,6,4,7,8,9,8,7,6,5,4,1,2,3,6,5,4])
N.Train()
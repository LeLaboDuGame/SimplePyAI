import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

color1 = '\033[92m'
color2 = '\033[94m'

print(
    f"{color1}You use the library SimplePyAI !\n{color2}Credit: LeLaboDuGame on Twitch -> https://twitch.tv/LeLaboDuGame")



class Neural_Network:
    """
    SIMPLEPYAI
    ----------
    Hey guy !
    I'm 15 years frensh dev !
    I present you my project of neural network !
    Just for the credit : LeLaboDuGame on Twitch -> https://twitch.tv/LeLaboDuGame
    You can use this library on all your project !

    HOW TO USE:
    -----------
        TO START:
            -To init you must send 'x' and 'y' list (or numpy array).
            'x' is your input and 'y' is your output.

            -You must put the 'layer' (default is [2, 3, 1] 2 is the input and 1 the output neurone).
            The 'layer' is a list represent all of your layers with the number of your neurone example:
            [4,                                     3,4,10,1,                                       100]
            (is for 4 neurone in the first layer)(some layers)(And the number of neurones in the output)

            -You can set the 'learning_rate' more is small more he understands, but it will take more repetition in
            training.

            -'reload_last_session' (default: False) is to reload the last session (if is the first sessions set to False plz)

        TO TRAIN:
            -'n_iter' is the number of train repetition

            -'show' (default: True) is to show how the graphique of your model

            -'save' (default: False) is to save your model (you can reload your model with the 'reload_last_session' in init

    Thank you very much to use this library !
    """

    def __init__(self, x, y, layers=None, learning_rate=0.1, reload_last_session=False):
        if layers is None:
            layers = [2, 3, 1]
        self.couche = layers
        self.learning_rate = learning_rate
        self.y = np.array(y)
        self.x = np.array(x)

        self.parametres = {}
        if reload_last_session:
            a_file = open("ia_parametres.pkl", "rb")
            self.parametres = pickle.load(a_file)
            a_file.close()

        else:
            # Initialistaion des dict W (weight) et B (Bias) avec les couches definie
            for c in range(1, len(layers)):
                self.parametres[f"W{c}"] = np.random.randn(layers[c], layers[c - 1])
                self.parametres[f"b{c}"] = np.random.randn(layers[c], 1)

    def forward_propagation(self, x):
        A = {"A0": x.T}

        C = len(self.parametres) // 2

        for c in range(1, C + 1):
            Z = self.parametres[f"W{c}"].dot(np.array(A[f"A{c - 1}"])) + self.parametres[f"b{c}"]
            A[f"A{c}"] = 1 / (1 + np.exp(-Z))
        return A

    def back_propagation(self, y, A):
        m = y.shape[1]
        C = len(self.parametres) // 2

        dZ = A[f"A{C}"] - y.T
        gradient = {}

        for c in reversed(range(1, C + 1)):
            gradient[f"dW{c}"] = 1 / m * np.dot(dZ, A[f"A{c - 1}"].T)
            gradient[f"db{c}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.parametres[f"W{c}"].T, dZ) * A[f"A{c - 1}"] * (1 - A[f"A{c - 1}"])
        return gradient

    def predict(self, x, with_pourcent=False):
        A = self.forward_propagation(np.array(x))
        if not with_pourcent:
            return A[f"A{len(A) - 1}"] > 0.5
        else:
            return A[f"A{len(A) - 1}"] > 0.5, A[f"A{len(A) - 1}"]

    def show(self, Loss, acc):
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        plt.plot(Loss)
        plt.subplot(1, 3, 2)
        plt.plot(acc)
        plt.subplot(1, 3, 3)

        C = len(self.parametres) // 2
        plt.scatter(self.x[:, 0], self.x[:, len(self.x[0]) - 1], c=self.y, cmap="summer")
        plt.show()

    def update(self, gradient):
        C = len(self.parametres) // 2

        for c in range(1, C + 1):
            self.parametres[f"W{c}"] = self.parametres[f"W{c}"] - self.learning_rate * gradient[f"dW{c}"]
            self.parametres[f"b{c}"] = self.parametres[f"b{c}"] - self.learning_rate * gradient[f"db{c}"]

    def log_loss(self, A, y):
        m = len(y)
        epsilon = 1e-15
        return 1 / m * np.sum(-y.T * np.log(A + epsilon) - (1 - y).T * np.log(1 - A + epsilon))

    def train(self, n_iter, show=True, save=False):
        acc = []
        Loss = []
        for episode in tqdm(range(n_iter)):
            # methode d'activation
            A = self.forward_propagation(self.x)
            gradients = self.back_propagation(self.y, A)
            self.update(gradients)
            if episode % 10 == 0:
                C = len(self.parametres) // 2
                Loss.append(self.log_loss(A[f"A{C}"], self.y))
                y_pred = self.predict(self.x)
                acc.append(accuracy_score(self.y.flatten(), y_pred.flatten()))

        # prediction
        y_pred = self.predict(self.x)
        print(f"Score de l'entrainement = {accuracy_score(self.y.flatten(), y_pred.flatten()) * 100}%")

        # montre la courbe de Loss montrant la diminution de la perte du model
        print(f"La perte est de : {Loss[len(Loss) - 1]} !")

        if save:
            a_file = open("ia_parametres.pkl", "wb")
            pickle.dump(self.parametres, a_file)
            a_file.close()

        if show:
            self.show(Loss, acc)
        return y_pred

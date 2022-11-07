Is a simple python library to use the concept of deep learning in your project (is verrrrrrryyyyyy simple for shur)

# SIMPLEPYAI
Hey guy !
I'm 15 years and a frensh dev !
I present you my project of neural network !
Just for the credit : LeLaboDuGame on Twitch -> https://twitch.tv/LeLaboDuGame
You can use this library on all your project !
### Now it's possible to choose an activation function, to do that you can refer at the "sigmoid" function and the "derivation_sigmoid" function

**The model use the sigmoïde activation function.**

#   HOW TO USE:
    
    
### ON FIRST:
    Download all of this library: pickle,matplotlib,sklearn,tqdm,numpy
    
## EXAMPLE:

    x = [[1], [0]]
    y = [[0], [1]]
    nn = Neural_Network(x, y, layers=[1, 64, 32, 64, 1], learning_rate=0.001, activation_function=sigmoid,
                        derivation=derivation_sigmoid, reload_last_session=False)
    nn.train(10000, show=True)
    print(nn.predict([[1], [0]]))  # predict values

## TO START:

    -To init you must send 'x' and 'y' list (or numpy array).
    'x' is your input and 'y' is your output.

    -You must put the 'layers' (default is [2, 3, 1] 2 is the input and 1 the output neurone).
    The 'layers' is a list represent all of your layers with the number of your neurone example:

    [4,                                     3,4,10,1,                                       100]
     ^                                          ^                                             ^ 
     |                                          |                                             |
    (is for 4 neurone in the first layer)(some layers)(And the number of neurones in the output)

    -You can set the 'learning_rate' more is small more he understands, but it will take more repetition in
    training.

    -'reload_last_session' (default: False) is to reload the last session (if is the first sessions set to False)

## TO TRAIN:

    -'n_iter' is the number of train repetition

    -'show' (default: True) is to show how the graphique of your model

    -'save' (default: False) is to save your model (you can reload your model with the 'reload_last_session'
    in init.

# Thank you very much to use this library !

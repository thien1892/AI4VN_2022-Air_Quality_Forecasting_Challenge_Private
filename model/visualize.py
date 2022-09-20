import matplotlib.pyplot as plt

def visualize_mdape(history, title, path_save):
    loss = history.history["mdape"]
    val_loss = history.history["val_mdape"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training mdape")
    plt.plot(epochs, val_loss, "r", label="Validation mdape")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("mdape")
    plt.legend()
    plt.savefig(path_save+'visualize_mdape.png')
    # plt.show()

def visualize_loss(history, title, path_save):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_save+'visualize_loss.png')
    # plt.show()
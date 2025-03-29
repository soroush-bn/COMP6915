from fastai.vision.all import *

def train_mnist():
    # ✅ Load MNIST dataset directly from Fastai
    path = untar_data(URLs.MNIST_SAMPLE)  # Small MNIST subset (3s and 7s)
    dls = ImageDataLoaders.from_folder(path, train='train', valid='valid', bs=64, num_workers=0)  # Set num_workers=0 for Windows

    # ✅ Define CNN Model using Fastai
    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # ✅ Train the model
    learn.fine_tune(5)

    # ✅ Evaluate on Test Set
    test_loss, test_acc = learn.validate()
    print(f"\n🔥 Test Accuracy: {test_acc:.4f}")

    # ✅ Save the trained model
    learn.export("mnist_fastai.pkl")

    # ✅ Load Model & Predict on a Sample Image
    learn_inf = load_learner("mnist_fastai.pkl")
    img = PILImage.create(path/'valid/3/9323.png')  # Load a test image
    preds, _, decoded_preds = learn_inf.predict(img)
    print(f"Predicted label: {decoded_preds}")
    img.show()

if __name__ == '__main__':  # ✅ Windows fix
    train_mnist()

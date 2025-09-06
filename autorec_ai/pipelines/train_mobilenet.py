from autorec_ai.train.custom_mobilenet import CustomMobileNet

if __name__ == "__main__":
    mobilenet = CustomMobileNet(freeze_backbone=False,  batch_size=128, lr=1e-4)
    mobilenet.train_and_evaluate(epochs=25)
    mobilenet.save()
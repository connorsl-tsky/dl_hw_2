
import time 
import os
import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


def export_stats(model, name: str, params: dict, num_runs: int, data: dict, random: int):
    """
    data can be
    {
    "test_x": testx,
    "testy": testy,
    valx
    valy
    trainx
    trainy
    }
    
    random is value of random state - just to remind me to have it the same, yeah?
    """
    for i in range(num_runs):
        start_time = time.time()
        history = model.train(data["train_x"], data["train_y"], data["val_x"], data["val_y"])
        end_time = time.time()

        l, acc, auc, f1, prec, rec = model.evaluate(data["test_x"], data["test_y"])

        # output results to file
        os.mkdir(name)  
        with open(name+"/"+name+"_"+i) as file:
            # print time
            file.write(f"TIME: {end_time-start_time}\n")

            # print history
            for k,v in history.history: # ?
                file.write(f"{k}: {v}\n")

            # print params
            for k,v in params:
                file.write(f"{k}: {v}\n")

            # print metrics
            file.write(f"LOSS: {l}]\n")
            file.write(f"ACCURACY: {acc}]\n")
            file.write(f"AUC: {auc}]\n")
            file.write(f"PRECISION: {prec}]\n")
            file.write(f"F1: {f1}]\n")
            file.write(f"RECALL: {rec}]\n")

            file.write(f"RANDOM STATE: {random}\n")

        # submission.csv
        yhat = np.array(model.model.predict(data["test_x"]))
        outdf = pd.DataFrame({"ImageId": data["test_y"], "Label": yhat})
        outdf.to_csv(name+"/submission_" + i + ".csv", index=False)

        # plot auc-roc curve
        display = RocCurveDisplay.from_predictions(
                data["test_y"].ravel(),
                yhat.ravel(),
                name="micro-average OvR",
                curve_kwargs=dict(color="darkorange"),
                plot_chance_level=True,
                despine=True,
            )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Title",
        )

        # TODO how to print multiple things on different plots?

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss)+1)
        plt.plot(epochs, loss, label=name+ " " + i + " Loss")
        plt.plot(epochs, val_loss, label=name + " " + i + " Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()


    return
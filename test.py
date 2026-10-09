
import time 
import os
import pandas as pd
import numpy as np
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import sklearn.preprocessing as skpre


def export_stats(model, name: str, params: dict, num_runs: int, data: dict, random: int, plot: bool=False, submission: bool=False):
    """
    data can be
    {
        "test_x": testx, - preprocessed
        "testy": testy, - one-hot encoded
        valx
        valy
        trainx
        trainy
    }
    
    random is value of random state - just to remind me to have it the same, yeah?
    
    what do we want to average
    time, l, acc, auc, f1, prec, rec, 
    """

    t_total_time = 0
    t_loss = 0
    t_acc = 0
    t_auc = 0
    t_f1 = np.array([0]*10, dtype=np.float64)
    t_prec = 0
    t_rec = 0

    for i in range(num_runs):
        start_time = time.time()
        # ah it's training the model num_runs times isn't it. without clearing it out. hm
        # https://stackoverflow.com/questions/40496069/reset-weights-in-keras-layer can save and load weights to reset it
        history = model.train(data["train_x"], data["train_y"], data["val_x"], data["val_y"])
        end_time = time.time()
        total_time = end_time-start_time

        l = 0
        acc = 0
        auc = 0
        f1 = 0
        prec = 0
        rec = 0
        if data["test"] is None:
            l, acc, auc, f1, prec, rec = model.evaluate(data["test_x"], data["test_y"])
            

        # output results to file
        if not os.path.exists(name):
            os.mkdir(name)  
        with open(f"{name}/{name}_{i}.txt", "w") as file:
            file.write("~~~~~RESULTS~~~~~\n\n")
            
            # print metrics
            file.write(f"TIME: {total_time:.3f} seconds\n")
            file.write(f"LOSS: {l:.3f}\n")
            file.write(f"ACCURACY: {acc:.3f}\n")
            file.write(f"AUC: {auc:.3f}\n")
            file.write(f"F1: {f1}\n")
            file.write(f"AVG F1: {np.average(f1):.3f}\n")
            file.write(f"PRECISION: {prec:.3f}\n")
            file.write(f"RECALL: {rec:.3f}\n")
            file.write(f"RANDOM STATE: {random}\n")

            file.write("\n~~~~~PARAMS~~~~~~\n\n")
            # print params
            for k,v in params.items():
                file.write(f"{k}: {v}\n")

            file.write("\n~~~~~HISTORY~~~~~\n\n")
            # print history
            for k,v in history.history.items():
                file.write(f"{k}:\n")
                for v_i in range(len(v)):
                    file.write(f"{v_i}: {v[v_i]}\n")
                file.write("\n")

        if data["test"] is not None:
            image_id = data["test"].index.to_numpy()+1

        if submission:
            # submission.csv
            # yhat = np.array(model.model.predict(data["test_x"]))
            yhat = np.array(model.model.predict(data["test"]))
            encoder = skpre.OneHotEncoder()
            categories = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
            encoder = encoder.fit(categories)
            # test_y = encoder.inverse_transform(data["test_y"]) # undo one-hot encoding
            submission_yhat = encoder.inverse_transform(yhat)
            submission_yhat = submission_yhat.reshape(submission_yhat.shape[0])
            # test_y = test_y.reshape(test_y.shape[0])
            # image_id = data["test"].index.to_numpy()+1
            # outdf = pd.DataFrame({"ImageId": test_y, "Label": submission_yhat})
            outdf = pd.DataFrame({"ImageId": image_id, "Label": submission_yhat})
            outdf.to_csv(f"{name}/submission_{i}.csv", index=False)

        if plot:
            # plots
            fig, ax = plt.subplots(1, 2)

            # plot auc-roc curve
            display = RocCurveDisplay.from_predictions(
                    data["test_y"].ravel(),
                    yhat.ravel(),
                    name="ROC",
                    curve_kwargs=dict(color="darkorange"),
                    plot_chance_level=True,
                    despine=True,
                    ax=ax[0]
                )
            _ = display.ax_.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="AUC-ROC Curve",
            )

            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1, len(loss)+1)
            ax[1].plot(epochs, loss, label=f"{name} {i} Loss")
            ax[1].plot(epochs, val_loss, label=f"{name} {i} Validation Loss")
            ax[1].set_title("LOSS")
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Loss')
            ax[1].legend()

            plt.show()

        t_total_time += total_time
        t_loss += l
        t_acc += acc
        t_auc += auc
        t_f1 += f1
        t_prec += prec
        t_rec += rec

    avg_total_time = t_total_time/num_runs
    avg_loss = t_loss/num_runs
    avg_acc = t_acc/num_runs
    avg_auc = t_auc/num_runs
    avg_f1 = t_f1/num_runs
    avg_prec = t_prec/num_runs
    avg_rec = t_rec/num_runs
    
    with open(f"{name}/{name}_AVG.txt", "w") as file:
        file.write(f"AVG TIME: {avg_total_time:.3f}\n")
        file.write(f"AVG LOSS: {avg_loss:.3f}\n")
        file.write(f"AVG ACCURACY: {avg_acc:.3f}\n")
        file.write(f"AVG AUC: {avg_auc:.3f}\n")
        file.write(f"AVG F1: {avg_f1}\n")
        file.write(f"AVG AVG F1: {np.average(avg_f1):.3f}\n")
        file.write(f"AVG PRECISION: {avg_prec:.3f}\n")
        file.write(f"AVG RECALL: {avg_rec:.3f}\n")

    return

if __name__ == "__main__":
    print("hello world")
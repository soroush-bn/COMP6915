from fastai.tabular.all import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# ✅ Load dataset correctly
digits = load_digits()
X, y = digits.data, digits.target

# ✅ Convert to DataFrame
df = pd.DataFrame(X)
df['label'] = y

# ✅ Split dataset correctly
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# ✅ Convert labels to category (Fastai requires categorical labels)
df_train['label'] = df_train['label'].astype(str)
df_test['label'] = df_test['label'].astype(str)

# ✅ Fastai processing
procs = [Categorify, Normalize]  # Normalize inputs correctly
dls = TabularDataLoaders.from_df(df_train, y_names="label", cont_names=list(df_train.columns[:-1]), procs=procs, bs=64)

# ✅ Define and train a better model
learn = tabular_learner(dls, layers=[128, 64], metrics=accuracy, loss_func=CrossEntropyLossFlat())  # Correct loss function

# ✅ Find best learning rate
learn.lr_find()

# ✅ Train with an optimized learning rate
learn.fit_one_cycle(10, 0.01)

# ✅ Validate performance
valid_loss, valid_acc = learn.validate()
valid_err = 1 - valid_acc
print(f"✅ Validation Error: {valid_err:.4f}")

# ✅ Test on separate dataset
dl_test = learn.dls.test_dl(df_test)
test_preds, _ = learn.get_preds(dl=dl_test)
test_acc = accuracy(test_preds, torch.tensor(df_test['label'].astype(int).values)).item()
test_err = 1 - test_acc

print(f"✅ Test Error: {test_err:.4f}")

# ✅ Save model
learn.export("final_mlp_model.pth")
print("✅ Model saved as 'final_mlp_model.pth'")

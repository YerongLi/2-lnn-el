from RuleLNN_nway import *

# train and val

df_train_val = pd.read_csv("train.csv")

features_train_val = np.array([np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_train_val.Features.values])

#to train a xor we need its truth table
X_train_val = torch.from_numpy(features_train_val).float()
print(X_train_val, X_train_val.shape)
#the target values for each row in the truth table (xor)
Y_train_val = torch.from_numpy(df_train_val.Label.values).float()
print(Y_train_val, Y_train_val.shape)
# mention_labels (cannot convert string explicitly)
mention_labels_train_val = df_train_val.Mention_label.values
print(mention_labels_train_val)

x_train, x_val, y_train, y_val, m_labels_train, m_labels_val = \
    train_test_split(X_train_val, Y_train_val, mention_labels_train_val, test_size=0.2,train_size=0.8, random_state=100)

# test
df_test = pd.read_csv("test.csv")
# df_train_val = df_train_val.loc[22:25]

features_test = np.array([np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_test.Features.values])

x_test = torch.from_numpy(features_test).float()
print(x_test, x_test.shape)
y_test = torch.from_numpy(df_test.Label.values).float()
print(y_test, y_test.shape)
m_labels_test = df_test.Mention_label.values
print(m_labels_test)

# Train PureNameLNN

# Sanity Check
model = ContextLNN(0.9, 2, False)
print(model(x_train, m_labels_train))

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

def evaluate(eval_model, x_eval, y_eval, m_labels_eval):
    eval_model.eval()
    with torch.no_grad():
        val_pred = eval_model(x_eval, m_labels_eval)
        loss = loss_fn(val_pred, y_eval)
        val_pred_ = val_pred > 0.5
        print("val loss", loss)
        prec, recall, f1, _ = precision_recall_fscore_support(y_eval, val_pred_, average='macro')
        print("f1 w/ 0.5 threshold", f1)
    return loss, f1, val_pred


best_pred = None
best_val_f1, best_val_loss = 0, 10000

for iter in range(200):

    model.train(True)
    optimizer.zero_grad()

    yhat = model(x_train, m_labels_train)
    loss = loss_fn(yhat, y_train)

    print("Iteration " + str(iter) + ": " + str(loss.item()))
    loss.backward()
    optimizer.step()

    val_loss, val_f1, val_pred = evaluate(model, x_val, y_val, m_labels_val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_f1 = val_f1
        best_pred = val_pred
        torch.save(model.state_dict(), "best_ContextLNN.pt")

# tune on val set

print("Val -- The best f1 is {} w/ naive threshold 0.5".format(best_val_f1))

best_tuned_threshold = 0.5
best_tuned_f1 = best_val_f1

for threshold_ in np.linspace(0.0, 1.0, num=1000):
    y_val_preds = best_pred >= threshold_
    prec, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_preds, average='macro')
    if f1 > best_tuned_f1:
        best_tuned_threshold = threshold_
        best_tuned_f1 = f1
print("Val -- After tuning, the best f1 is {} w/ threshold {}".format(best_tuned_f1, best_tuned_threshold))


bestModel = ContextLNN(0.9, 2, False)
bestModel.load_state_dict(torch.load("best_ContextLNN.pt"))
bestModel.eval()

with torch.no_grad():
    test_pred = bestModel(x_test, m_labels_test)
    test_pred = test_pred >= best_tuned_threshold
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='macro')
    print("Test -- f1 is {} w/ threshold {}".format(f1, best_tuned_threshold))
    print("prec, recall, f1", prec, recall, f1)


for name, mod in bestModel.named_children():
    print("========={}=========".format(name))
    if 'or' in name.lower():
        print(mod.AND.cdd())
    else:
        print(mod.cdd())
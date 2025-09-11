import matplotlib.pyplot as plt
from src.dataset import generate_data
from src.model import SimpleNN
from sklearn.linear_model import LinearRegression


X, y = generate_data(200)


nn = SimpleNN(input_dim=1, hidden_dim=10, output_dim=1, lr=0.01)
losses = nn.train(X, y, epochs=2000)


y_pred_nn = nn.forward(X)


lr = LinearRegression()
lr.fit(X, y)
y_pred_lr = lr.predict(X)


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Training Loss")

plt.subplot(1,2,2)
plt.scatter(X, y, label="True Data", alpha=0.5)
plt.plot(X, y_pred_nn, label="NN Prediction", color="red")
plt.plot(X, y_pred_lr, label="Linear Regression", color="green")
plt.legend()
plt.title("Regression Comparison")

plt.show()

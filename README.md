
#  Simple Neural Network Regression



##  Project Structure
```

simple-nn-regression/
│── main.py                # Runs training + visualization
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── src/
│   ├── dataset.py         # Data generation (noisy sine wave)
│   └── model.py           # SimpleNN implementation

````

---

##  Getting Started

###  Clone the repo
```bash
git clone https://github.com/hanashamsa/simple-nn-regression.git
cd simple-nn-regression
````

###  Install dependencies

```bash
pip install -r requirements.txt
```

###  Run the project

```bash
python main.py
```

---

##  Results

The script produces two plots:

1. **Training Loss** (loss decreasing across epochs)
2. **Regression Comparison**

   * Blue: True noisy sine data
   * 🔴 Red: Neural Network prediction
   * 🟢 Green: Linear Regression prediction

---



##  Future Improvements

* Add deeper architectures (multi-hidden layers)
* Try different activation functions (tanh, sigmoid, etc.)
* Add early stopping & regularization
* Implement mini-batch training

---


 Developed with ❤️ using **NumPy, scikit-learn, and Matplotlib**.


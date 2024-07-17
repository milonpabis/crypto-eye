# Overview
---

The main goal of the project was to create classificators that will predict whether the price of the **Bitcoin** will grow on beginning of the next day.<br><br>
**Important!**<br><br>
I did not want to create any model that would predict the price of the Bitcoin itself, due to the **Extrapolation** and **Unstationarity** of this currency.<br>

---
1. First, I went through **Feature Engineering** process, looking for the best technical stock metrics that would help the predictors make precise predictions.
   I used metrics like: <i>**RSI, MACD, CCI, Stochastic Oscilator**</i> on different periods, and found the best working ones.

2. The next step was to create some sort of simulation (**Rolling and Anchored Walk Forward**) to evaluate the models without the leakage of the future data.

3. I evaluated different models, hyperoptimized them and picked the best 3 of them:
   - GradientBoost
   - AdaBoost
   - RandomForest

4. I implemented **Database** storing the model predictions and performance over time. So simply just a little version of model monitoring.

5. Created a simple web app with daily results and performance tracking.

---

# Information
---

The models are not perfect, they were definitely performing better in the past, however they can always be enhanced, or some new models can be introduced here, especially
knowing that the app is flexible and ready to add a new model, or easily change hyperparameters of the existing ones.

---

# App Preview
---
![ce1](https://github.com/user-attachments/assets/50f684fe-e4bd-410d-8064-6dbbfa9b1339)
![ce2](https://github.com/user-attachments/assets/9ccb606a-c888-4162-aa09-9489cd1bd561)
![ce3](https://github.com/user-attachments/assets/a669dbf2-0da3-459a-beae-edc0d4c6b5f5)
![ce4](https://github.com/user-attachments/assets/83ef3549-55e2-4ee6-8851-c4ba1559431f)
![ce5](https://github.com/user-attachments/assets/2fd38f38-917b-4ba4-a42c-27b5ae9ffa04)


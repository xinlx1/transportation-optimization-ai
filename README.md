🚦 Transportation-Optimization-AIThis repository contains the deep learning framework I developed during the Google Collaboration Project. The goal was simple but ambitious: use AI to tackle the mess that is large-scale urban traffic and energy waste.

🧠 The ChallengeTraffic isn't just about cars; it's a massive optimization problem involving unpredictable flow and energy consumption. My task was to build a model that doesn't just "predict" but stays stable under different scales of data.

🛠️ What I DidInstead of just throwing data at a standard model, I spent a lot of time "under the hood":Architecture Tuning: Experimented with different neural network structures in TensorFlow to find the sweet spot for transportation data.The 15% Boost: By implementing custom hyperparameter search and regularization (to stop the model from just memorizing the noise), I managed to push the accuracy and training stability up by 15%.Stress Testing: I didn't just test it on one dataset; I ran benchmarks across multiple training configurations to make sure it actually works in the real world, not just on my machine.

💻 Tech StackCore: Python, TensorFlow Focus: Hyperparameter Optimization, Regularization, Benchmarking 

📈 Key ResultsSuccessfully predicted energy forecasting with significantly lower variance.Improved model convergence speed through better weight initialization and tuning.

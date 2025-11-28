# 🎯 DQN Project & Multi‑Environment RL Suite

This repository contains training and evaluation pipelines for **multiple RL environments** using **DQN and other algorithms**, supported by Stable-Baselines3, Gym/Gymnasium, structured episode termination logic, automated diagnostics, and (optional) GitHub workflow automation.

---

## 🚀 Highlights

- Supports **multiple environments** (e.g., CartPole, custom envs, wrappers).
- Modular training pipeline under `scripts/`.
- Structured termination classification:
  - **SOLVED** — truncated and reached max steps  
  - **GOOD_RUN** — terminated normally with high score (≥195)  
  - **TIME_LIMIT** — truncated early  
  - **FAIL** — terminated early (fell / out of bounds)  
  - **UNKNOWN** — non-standard  
- **Best-run video recording** (saves only SOLVED episodes).
- Loss/reward plots, evaluation summaries, TensorBoard logs.
- Optional **GitHub Actions workflows** under `.github/workflows/`.

---

## 📁 Repository Structure

```
/                                       # Root
├── scripts/                            # Training & evaluation pipelines
│   ├── train_dqn_cartpole.py
│   ├── evaluate.py
│   └── ... (other envs / algorithms)
│
├── envs/                               # (Optional) custom envs or configs
│
├── models/                             # Saved models, plots, videos
│
├── results/                            # Evaluation summaries
│
├── .github/workflows/                  # CI / training automation
│   └── run_training.yml
│
├── requirements.txt
└── README.md
```

---

## 🧰 Installation

```
pip install -r requirements.txt
```

---

## 🎮 Training Example

```
python scripts/train_dqn_cartpole.py     --total_steps 300000 --stage_size 50000 --lr 5e-4
```

Outputs:
- model checkpoints  
- reward + loss plots  
- videos (only on SOLVED)  
- evaluation results  

---

## 🧪 Evaluation

```
python scripts/evaluate.py     --model_path models/dqn_cartpole_v1.zip     --env CartPole-v1     --episodes 20
```

---

## 🎥 Episode Outcome Classification

| Outcome      | Condition |
|--------------|-----------|
| SOLVED       | truncated == True **and** steps == max_steps |
| GOOD_RUN     | terminated == True **and** steps ≥ 195 |
| TIME_LIMIT   | truncated == True **and** steps < max_steps |
| FAIL         | terminated == True **and** steps < 195 |
| UNKNOWN      | Anything else |

---

## 🤖 GitHub Workflows (Optional)

Your `.github/workflows/` directory may include automation such as:

- Auto-training on push  
- Scheduled evaluation  
- Artifact uploads (models, plots)  
- Notebook execution  

Add a badge:

```
![Workflow](https://github.com/ai-autonomous/dqn-project/actions/workflows/run_training.yml/badge.svg)
```

---

## 🤝 Contributing

Contributions are welcome:
- Add new environments
- Add new RL algorithms (DDQN, Dueling, PER, PPO, etc.)
- Improve workflow automation
- Add documentation / examples

---

## 📄 License

Add your preferred open-source license in `LICENSE`.

---

Enjoy experimenting with reinforcement learning across multiple environments!

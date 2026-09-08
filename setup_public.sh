# clone the repo
git clone git@github.com:StanfordMIMI/HeadEntropy.git
cd HeadEntropy

# check
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv ~/he --python 3.12
source ~/he/bin/activate
uv pip install vllm --torch-backend=cu129 \
  --extra-index-url https://wheels.vllm.ai/0.26.0/cu129
uv pip install matplotlib scipy pandas scikit-learn statsmodels \
  transformers datasets torchmetrics tabulate tqdm ipdb accelerate shap xgboost torchmetrics --torch-backend=cu129

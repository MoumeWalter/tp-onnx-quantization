# TP ONNX : export, benchmark et quantification

- Entraînement d'un CNN (FashionMNIST, CIFAR-10) avec PyTorch
- Export ONNX, benchmark PyTorch vs ONNX Runtime (batch 1, 8, 32, 128)
- Quantification dynamique et statique (int8), comparaison taille / accuracy / vitesse
- Export web avec onnxruntime-web (dossier `web/`)

## Lancer
pip install -r requirements.txt
jupyter notebook TP_onnx_quantization.ipynb

## Web
cd web && python -m http.server 8000  # puis http://localhost:8000
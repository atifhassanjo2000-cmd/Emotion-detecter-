# Deploy on Render

1. In Colab, download `multimodal_emotion_stress_pro.pth` (Files panel > right-click > Download)
   and put it next to `app.py`. (Optional: without it the app uses the pretrained face model only.)
2. Push this folder to a GitHub repo.
3. Render dashboard > New > Blueprint > pick the repo (it reads `render.yaml`).
4. Wait for the build (~10 min first time). Open the URL and allow camera access (HTTPS is required for webcams; Render provides it).

Local test: `pip install -r requirements.txt && python app.py` then open http://localhost:10000

Notes: 512 MB plans will crash (torch + the HF model need ~1.5 GB). Use Standard (2 GB).

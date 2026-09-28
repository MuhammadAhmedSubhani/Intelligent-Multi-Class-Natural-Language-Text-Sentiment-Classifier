# Integration

Put these files in your existing project root:

```text
project-root/
├── app.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── src/
├── models/
├── data/
├── results/
└── reports/
```

Install Flask:

```powershell
pip install flask
```

Run from the project root:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Do not move `app.py` into `src/`. The backend imports your existing `src/preprocessing.py` and loads the existing model files from `models/`.

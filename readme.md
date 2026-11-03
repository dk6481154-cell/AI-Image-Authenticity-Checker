
# 🔍 AI Image Authenticity Checker

An image forensics and AI image detection web application built using Python, Streamlit, OpenCV, PyTorch, and Hugging Face Transformers.

The application analyzes uploaded images and provides indicators that may help users investigate whether an image is AI-generated or potentially manipulated.

## 🚀 Features

- Upload JPG, JPEG, PNG, and WEBP images.
- AI-generated image classification using a pretrained model.
- Image metadata and EXIF inspection.
- Error Level Analysis (ELA).
- Image dimensions and file size analysis.
- SHA-256 image hashing.
- AI model prediction scores.
- SQLite analysis history.
- Downloadable PDF analysis reports.
- CSV export of analysis history.
- Interactive Streamlit dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend |
| Streamlit | Web interface |
| OpenCV | Image processing |
| Pillow | Image manipulation |
| NumPy | Numerical processing |
| PyTorch | Deep learning |
| Hugging Face Transformers | Pretrained image classifier |
| Pandas | Data management |
| SQLite | Database |
| ReportLab | PDF reports |

## 📁 Project Structure

```text
AI-Image-Authenticity-Checker/
│
├── app.py
├── requirements.txt
├── README.md
└── image_history.db
```

The SQLite database is created automatically when the application runs.

## ⚙️ Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Image-Authenticity-Checker.git
```

### Step 2: Open Project Directory

```bash
cd AI-Image-Authenticity-Checker
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Linux/Mac:

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🤖 AI Detection Model

Default Hugging Face model:

`umm-maybe/AI-image-detector`

The model is downloaded automatically when the detection feature is first used.

You can change the model by setting the `HF_MODEL_ID` environment variable.

Windows PowerShell:

```powershell
$env:HF_MODEL_ID="umm-maybe/AI-image-detector"
```

Linux/Mac:

```bash
export HF_MODEL_ID="umm-maybe/AI-image-detector"
```

## 🌐 Deployment on Streamlit Community Cloud

1. Upload your project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select your repository.
5. Select `app.py` as the main file.
6. Click Deploy.

Make sure the deployment environment has enough memory to load PyTorch and the pretrained model.

## 📊 How It Works

1. The user uploads an image.
2. The application extracts image properties and metadata.
3. ELA creates a recompressed comparison visualization.
4. A pretrained classifier estimates the image category.
5. Results are displayed in the dashboard.
6. Analysis information is saved in SQLite.
7. The user can download a PDF report.

## ⚠️ Important Limitations

- AI detection results are estimates, not proof.
- Model performance depends on its training data and image type.
- Metadata can be removed or modified.
- ELA can show differences caused by normal compression.
- A missing EXIF record does not mean an image is fake.
- The application cannot guarantee the authenticity of an image.

Use independent evidence for high-stakes verification.

## 👨‍💻 Author

Your Name

## 📜 License

This project is available under the MIT License.

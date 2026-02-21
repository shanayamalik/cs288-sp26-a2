# Quick Start: Running Part 4 on Google Colab

## ✅ Your code is now on GitHub!
**Repository:** https://github.com/shanayamalik/cs288-sp26-a2

---

## 🚀 How to Run on Google Colab

### Step 1: Open the Notebook in Colab

**Option A: Direct Link (Easiest)**
1. Go to: https://colab.research.google.com/github/shanayamalik/cs288-sp26-a2/blob/main/CS288_A2_Part4_Training.ipynb
2. The notebook will open directly in Colab!

**Option B: Manual Upload**
1. Download the notebook from GitHub
2. Go to https://colab.research.google.com
3. Click `File` → `Upload notebook`
4. Select the downloaded notebook

---

### Step 2: Enable GPU
1. In Colab, click `Runtime` → `Change runtime type`
2. Select `Hardware accelerator: GPU`
3. Choose `T4 GPU` (or whatever is available)
4. Click `Save`

---

### Step 3: Run the Notebook

**Option A: Run All at Once**
- Click `Runtime` → `Run all`
- Wait for completion (~15-30 minutes for "small" config)

**Option B: Run Cell by Cell**
- Press `Shift+Enter` to run each cell
- Monitor progress as you go

---

### Step 4: Download Your Results

The notebook will automatically:
1. ✅ Generate `finetuned_predictions.json`
2. ✅ Generate `prompting_predictions.json`
3. ✅ Download both files to your computer

Look in your **Downloads** folder for the JSON files!

---

## ⚙️ Configuration Options

In Cell 5 (Configuration section), you can adjust:

```python
CONFIG_NAME = "small"  # Change to "quick", "small", or "medium"
```

### Recommended Settings:

| Config | Time | Model Size | Best For |
|--------|------|------------|----------|
| **quick** | 5-10 min | ~1M params | Testing the pipeline |
| **small** | 15-30 min | ~10M params | **Submission (recommended)** |
| **medium** | 1-2 hours | ~50M params | Best possible accuracy |

---

## 📊 What to Expect

After running, you'll see:
- Pre-training loss decreasing
- Generated text samples
- Fine-tuned model accuracy on dev set
- Prompting model accuracy on dev set
- Two JSON files downloaded

**Expected accuracy ranges:**
- Fine-tuned: 35-50%+
- Prompting: 30-55%+ (ideally 2% better than fine-tuned)

---

## 🐛 Troubleshooting

### "Out of Memory" Error
- Reduce batch size in Cell 5:
  ```python
  CONFIG["batch_size"] = 16  # or 8
  ```
- Or use "quick" config

### "Repository not found"
- Make sure your repo is **public** 
- Or update Cell 2 with your GitHub token:
  ```python
  !git clone https://YOUR_TOKEN@github.com/shanayamalik/cs288-sp26-a2.git
  ```

### "Module not found"
- Re-run Cell 3 (Install Dependencies)
- Check that Cell 4 (Test Imports) passes

### Slow Training
- Normal! GPU training takes time
- Keep the Colab tab open
- Consider using Colab Pro for longer sessions

---

## 📝 Files You Need to Submit

After the notebook finishes:
1. ✅ `finetuned_predictions.json` (from your Downloads folder)
2. ✅ `prompting_predictions.json` (from your Downloads folder)

Submit both files according to your assignment instructions.

---

## 🔗 Useful Links

- **Your GitHub Repo:** https://github.com/shanayamalik/cs288-sp26-a2
- **Open in Colab:** https://colab.research.google.com/github/shanayamalik/cs288-sp26-a2/blob/main/CS288_A2_Part4_Training.ipynb
- **Google Colab:** https://colab.research.google.com

---

## 💡 Tips

1. **First time?** Start with `CONFIG_NAME = "quick"` to test (~5 min)
2. **For submission:** Use `CONFIG_NAME = "small"` (~15-30 min)
3. **Keep Colab active:** Don't close the tab while running
4. **Monitor progress:** Watch the output as it trains
5. **Save results:** The notebook auto-downloads the JSON files

---

## Need Help?

If something goes wrong:
1. Check the error message in the notebook
2. Try restarting: `Runtime` → `Restart runtime`
3. Re-run from the beginning: `Runtime` → `Run all`

Good luck! 🚀

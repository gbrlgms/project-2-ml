![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Deep Impressions: Training a Neural Network to Recognize Painters

**Project 2 | Machine Learning: From Data to Predictions**

Gabriel Gomes |  Ironhack Data Science & Machine Learning Bootcamp 

How well can a neural network tell apart painters with very similar styles from adjacent art movements? This project trains an image classifier to name the painter of a painting, choosing among Monet, Renoir, Morisot, Manet, Cézanne, and Van Gogh.

## Result at a glance

| | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline (always answer "Van Gogh") | 0.335 | 0.084 |
| **Final model, test set (812 paintings)** | **0.873** | **0.806** (95% CI 0.766 to 0.840) |
| Final model, validation set | 0.878 | 0.820 |

The final model is a MobileNetV2 pretrained on ImageNet, with its last four blocks fine-tuned on the paintings. On paintings it had never seen, it names the right painter about 7 times in 8. The validation score falls inside the test interval, so the estimate was honest. Its weakest point is Manet: about half of its "Manet" labels are wrong.

---

## 1. The question

**What is predicted:** given an image of a painting, which of six painters made it.

**Why these six:** they form a continuum of neighboring styles, which is what makes the task hard.

| Style | Painters | Note |
|---|---|---|
| Realism → Impressionism | Manet | The bridge: never exhibited with the group, but inspired it |
| Impressionism | Monet, Renoir, Morisot | The core of the movement |
| Post-Impressionism | Cézanne, Van Gogh | Kept the bright color, pushed toward structure (Cézanne) and emotion (Van Gogh) |

**For whom, and what they would do with it:** catalogers with unlabeled or doubtful entries, museum visitors who want a hint about what they are looking at, and art students training their eye. In each case the model proposes a painter and a person confirms or corrects it. It is a shortlist generator, not an authority, so a wrong suggestion is caught in review. These are the scenarios the project was designed for, not users it was tested with.

**Target and positive class:** the target is the painter's name, a categorical variable with 6 classes. This is multi-class classification, so there is no single positive class. Each painter is treated as the positive class in turn (one-vs-rest) when computing per-class precision, recall, and F1.

## 2. The data

**Source**: WikiArt, an online art encyclopedia in which each artwork is labeled with its artist, style, and genre. The images were downloaded with the open-source [WikiART Crawler](https://github.com/asahi417/wikiart-image-dataset) (`wikiartcrawler`), a Python library that fetches images from WikiArt through its API, one artist at a time, and caches them on the local machine. For each of the six painters, the crawler's `get_painting` call returns the paths of that painter's downloaded images, which is where the artist label comes from.

**License**: only paintings in the public domain were kept. The images are downloaded from WikiArt onto the user's own machine and remain subject to WikiArt's terms of use, so they are not included in this repository. The images are reproductions of varying quality, and the labels (the artist each image was requested for) come from the site, so they are useful but not perfect.

**What one row is:** one painting image.

| Column | Type | Role |
|---|---|---|
| `path` | text | File location of the image, the model's input |
| `artist` | text | The painter, **the target** |
| `title` | text | Metadata only, never used as an input |

**Size and balance:** 5,770 images. The classes are strongly imbalanced, at roughly 8 to 1 between the largest and the smallest.

| Painter | Images | Share |
|---|---:|---:|
| Van Gogh | 1,929 | 33.4% |
| Renoir | 1,412 | 24.5% |
| Monet | 1,365 | 23.7% |
| Cézanne | 587 | 10.2% |
| Morisot | 242 | 4.2% |
| Manet | 235 | 4.1% |

**Traps found and how they were handled**

- **The same work appears more than once.** Painters repeated scenes (Monet's haystacks, Van Gogh's sunflowers), and the dataset also contains sketches and rescans. If one version lands in training and another in the test set, the model is tested on a scene it has already seen and the score looks better than it should. Near-duplicates were found with perceptual hashing (`imagededup`) and removed. **[Add the number of images or groups removed.]**
- **The title can give the answer away** (*Water Lilies*, *Sunflowers*) and would not exist for a new, unlabeled painting, so it is never fed to the model.
- **Class imbalance** is handled with a class-weighted loss and a metric that weighs every painter equally.

**Split:** 70% training, 15% validation, 15% test (812 test images), stratified by painter and saved as CSV files so that every notebook uses exactly the same split.

## 3. Metric and baseline

**Main metric: macro-averaged F1.** It averages the F1 score of the six painters, so Manet and Morisot (about 4% of the data each) count as much as Van Gogh (33%). Plain accuracy would reward a model that favors the large classes.

**Which mistake costs more?** In a catalog, a confidently wrong attribution misleads the reader more than an honest "not sure", so the evaluation also looks at per-painter recall, the confusion matrix, and the model's confidence on its errors. No painter's misclassification is more costly than another's, so the metric is not weighted toward precision or recall for any particular class.

**Baseline to beat:** a `DummyClassifier` that always predicts the most frequent painter.

| Baseline | Accuracy | Macro-F1 |
|---|---:|---:|
| Most frequent class, 5-fold cross-validation | 0.335 ± 0.001 | 0.084 ± 0.000 |
| Most frequent class, test set | 0.335 | 0.084 |
| Guess by class share ("stratified"), test set | 0.246 | 0.166 |

The always-"Van Gogh" model is right a third of the time but scores zero on five of the six painters. That gap between accuracy and macro-F1 is why accuracy is only reported next to the baseline.

## 4. Models compared

All models use the same splits, the same augmentation (random horizontal flip, rotation of up to ±18°, and zoom of up to 10%, applied only during training), the same class-weighted cross-entropy loss, the AdamW optimizer, 224×224 inputs, and **early stopping on the validation loss** (patience 5, best weights restored). The framework is PyTorch.

| Model | What it is | Trainable parameters | Validation macro-F1 | Validation accuracy |
|---|---|---:|---:|---:|
| Baseline | Always the most frequent painter | n/a | 0.084 | 0.335 |
| **CNN from scratch** | 3 convolution blocks (32, 64, 128 channels), global average pooling, dropout 0.4 | about 94,000 | 0.51 | 0.64 |
| **MobileNetV2, frozen** | Pretrained on ImageNet; only a new final layer is trained (learning rate 1e-3) | 7,686 of 2.23 million | 0.76 | 0.835 |
| **MobileNetV2, fine-tuned** | The same, with the last 4 of 19 feature blocks also trained (learning rate 1e-5); BatchNorm kept frozen | about 1.5 million | **0.82** | 0.878 |

**What is transfer learning here?** ImageNet is a collection of about 1.2 million photographs in 1,000 everyday categories. A network pretrained on it has never seen art, but it already knows how to see edges, textures, and patterns. With only a few thousand paintings (a few hundred for Manet and Morisot), reusing those features works far better than learning them from nothing.

**What the comparison showed**

- The scratch CNN learned coarse cues such as color and stroke boldness: it sent 45 of 83 Cézanne paintings to Van Gogh.
- Pretrained features lifted macro-F1 from 0.51 to 0.76, and fine-tuning added 0.06 more, with gains for all six painters (largest for Manet, +0.12, and Morisot, +0.09).
- Fine-tuning overfits quickly (training macro-F1 reached about 0.94 against about 0.81 on validation), which is why early stopping was used.

## 5. Final result against the baseline

The test set was evaluated once, after the model, the stopping point, and the settings had been chosen.

| | Accuracy | Macro-F1 | Loss |
|---|---:|---:|---:|
| Validation | 0.878 | 0.820 | 0.403 |
| **Test** | **0.873** | **0.806** | 0.397 |
| Baseline, most frequent (test) | 0.335 | 0.084 | n/a |
| Baseline, stratified (test) | 0.246 | 0.166 | n/a |

A bootstrap of the test set gives a 95% interval for the macro-F1 of **0.766 to 0.840**, and the validation estimate (0.820) lies inside it, so there is no sign of leakage or of overfitting to the validation set. The interval is fairly wide because Manet and Morisot have only about 33 and 35 test images.

**Per-painter results on the test set**

| Painter | Test images | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Van Gogh | 272 | 0.96 | 0.92 | 0.94 |
| Renoir | 198 | 0.94 | 0.89 | 0.91 |
| Monet | 191 | 0.88 | 0.85 | 0.86 |
| Cézanne | 83 | 0.78 | 0.84 | 0.81 |
| Morisot | 35 | 0.68 | 0.80 | 0.74 |
| Manet | 33 | 0.49 | 0.70 | 0.58 |

### Where it goes wrong

- **Errors happen between painters who share a technique.** Ten Renoir paintings are labeled Monet (light and loose brushwork, painted side by side). Manet and Morisot get swapped (both paint loose figures), and so do Van Gogh and Cézanne (both use strong color). Monet's errors are spread across every other painter, which fits his role as the reference point of the style.
- **Manet is the catch-all.** He is predicted 47 times for 33 real paintings, so about half of the "Manet" labels are wrong. His style ranges from flat, dark early work to loose outdoor scenes, so the model uses his name for ambiguous paintings. Morisot shows the same effect, more mildly.
- **The model is overconfident.** Even its wrong predictions average a confidence of about 0.7, so a high probability is not a guarantee.

## 6. Limitations

- **Closed set:** the model must choose one of the six painters, so a painting by Degas, Pissarro, or Sisley will become a confident Monet or Renoir.
- **Reproductions only:** it was trained and tested on clean WikiArt images. Photos of framed paintings, sketches, drawings, and heavy crops are outside what it has seen.
- **Small test classes:** the Manet and Morisot figures could shift by several points on a different sample.
- **Interpretation:** the explanation that the model reads brushwork and texture, and not only color, is inferred from the error patterns and has not been proven (for example with saliency maps).
- **Not for authentication or valuation:** 87% is not enough, and the model sees broad style, not the evidence an expert uses.

**With more time:** calibrate the probabilities and choose the confidence cutoff on validation data; improve Manet and Morisot with tuned class weights and stronger augmentation; test on museum open-access collections and photos of paintings; try higher-resolution inputs and other backbones (for example EfficientNet, or CLIP embeddings with a logistic regression); and unfreeze fewer blocks with more regularization.

---

## 7. How to run it

### Repository layout

```
project-2-ml/
├── README.md
├── app.py          # Streamlit demo
├── data/split      # train.csv, val.csv, test.csv (created by the notebooks)
├── data/raw        # images, one folder per data/raw (not included)
│   ├── monet/
│   ├── van gogh/
│   └── ...
├── models      # stored PyTorch models
└── notebooks/                   
```

### 1. Environment

Python 3.14 in a clean virtual environment is recommended. 

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pandas numpy scikit-learn matplotlib pillow \
            imagededup networkx streamlit jupyter ipykernel
```

On Apple Silicon the notebooks use the GPU through PyTorch's MPS backend when it is available. If an operation is unsupported, start Jupyter with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### 2. Data

The images are downloaded with the [WikiART Crawler](https://github.com/asahi417/wikiart-image-dataset). It needs an internet connection and caches the files in `~/.cache/wikiartcrawler` by default.

1. Install the crawler:

   ```bash
   git clone https://github.com/asahi417/wikiart-image-dataset
   cd wikiart-image-dataset
   pip install .
   ```

2. Download the images of the six painters:

   ```python
   from wikiartcrawler import WikiartAPI, available_artist

   print(available_artist())          # lists the artist aliases the API knows, e.g. 'paul-cezanne'

   api = WikiartAPI()
   ARTISTS = [...]                    # [the six aliases you used]
   paths = {a: api.get_painting(a) for a in ARTISTS}   # local file paths per painter
   ```

3. Arrange the images as `data/raw/<artist>/<image>`, for example `data/raw/van gogh/starry_night.jpg`, and keep only the paintings that passed the public-domain check.

### 3. Notebooks

Run them in order:

1. `01_eda_and_baseline.ipynb`
2. `02_pipelines_and_models.ipynb`
3. `03_evaluation_and_interpretation.ipynb`

The final model is saved as `model_mobilenet_tuned.pt`: the weights together with the class list, the best epoch, and the training history.

### 4. The demo app

```bash
streamlit run app.py
```

Upload a painting and the app shows the predicted painter with its confidence, the probabilities for all six painters. Inference runs on the CPU.

The page uses the fonts of the project's title slide, loaded from Google Fonts, so it needs an internet connection to look as intended. For black accents and a light theme regardless of the system setting, add this to `.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#111111"
backgroundColor = "#F1EEEA"
secondaryBackgroundColor = "#E8E4DE"
textColor = "#111111"
```

### Reproducibility

Splits and bootstrap intervals use a fixed random seed (42). Training results can still vary slightly between runs because of random initialization and augmentation, and the reported numbers come from a single trained model.

---

## Credits

- Images and labels: [WikiArt](https://www.wikiart.org), downloaded with the [WikiART Crawler](https://github.com/asahi417/wikiart-image-dataset) library by asahi417.
- Model: MobileNetV2 (Sandler et al., 2018) with ImageNet pretraining (Deng et al., 2009), through `torchvision`.
- Near-duplicate detection: `imagededup`.
- Built as Project 2 of the Ironhack Data Science & Machine Learning Bootcamp.
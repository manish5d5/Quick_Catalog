# 👕 QuickCatalog - Smart Clothing Recognition

**QuickCatalog** is an AI-driven tool designed to help small and medium clothing shops automatically **recognize, tag, and record garments**.  
Simply upload or capture an image — the system predicts **Article Type**, **Base Colours**, and **Usage Type**, powered by **TensorFlow MobileNet** and an elegant **Streamlit interface**.

---

## 🚀 Features

- 🧠 **AI-Powered Recognition** using fine-tuned MobileNetV2  
- 📸 Upload or capture images in real-time  
- 🎨 Detects top-2 **Base Colours** with confidence levels  
- 🏷️ Predicts **Article Type** and **Usage Context**  
- 🖋️ Editable predictions for human confirmation  
- 💾 Simple, intuitive interface built in Streamlit  

---

## 🧩 Model & Label Details

### **Article Types (2 total)**
| Index | Name     |
| ------ | -------- |
| 0 | Shirts |
| 1 | Tshirts |

### **Base Colours (20 total)**
| Index | Colour |
| ------ | ------- |
| 0 | Beige |
| 1 | Black |
| 2 | Blue |
| 3 | Brown |
| 4 | Charcoal |
| 5 | Cream |
| 6 | Green |
| 7 | Grey |
| 8 | Lavender |
| 9 | Maroon |
| 10 | Navy Blue |
| 11 | Off White |
| 12 | Olive |
| 13 | Orange |
| 14 | Peach |
| 15 | Pink |
| 16 | Purple |
| 17 | Red |
| 18 | White |
| 19 | Yellow |

### **Usage Types (3 total)**
| Index | Usage |
| ------ | ------- |
| 0 | Casual |
| 1 | Formal |
| 2 | Sports |

---

## 🧠 Model Information

| Parameter | Value |
| ---------- | ------ |
| **Base Architecture** | MobileNetV2 |
| **Input Size** | 224 × 224 × 3 |
| **Framework** | TensorFlow / Keras |
| **Outputs** | Multi-head (Article Type, Base Colour, Usage) |
| **File** | `mobilenet_fashion_best.h5` |

---

## 🛠️ Requirements

| Library                    | Version   | Purpose                                                              |
| -------------------------- | --------- | -------------------------------------------------------------------- |
| **streamlit**              | ≥1.36.0   | For creating the interactive web app interface                       |
| **tensorflow-cpu**         | 2.15.0    | Loads and runs the trained MobileNetV2 model efficiently on CPU      |
| **numpy**                  | ≥1.24.0   | Numerical computations and preprocessing for image arrays            |
| **pillow (PIL)**           | ≥10.0.0   | For handling image uploads, resizing, and color conversions          |
| **opencv-python-headless** | ≥4.9.0.80 | Additional image preprocessing support (headless = no GUI)           |
| **pandas**                 | ≥2.1.0    | For structured data operations, useful for future dataset extensions |
| **jsonschema**             | ≥4.19.0   | To validate and manage JSON-based label mappings                     |
| **matplotlib**             | ≥3.8.0    | For optional visualization (charts, previews, etc.)                  |
| **scikit-learn**           | ≥1.3.0    | Useful for scaling, metrics, or later ML extensions                  |

To install all dependencies:
```bash
pip install -r requirements.txt


💡 Future Plans
🧾 Integrate database for saving recognized clothing data
📱 Add mobile camera optimization
🛍️ Batch image uploads for faster cataloging
☁️ Deploy to Streamlit Cloud or Hugging Face Spaces

🧠 Technical Tags
Computer Vision, Retail Automation, Deep Learning, Streamlit, TensorFlow, Image Classification, Smart Inventory


👨‍💻 Author
Developed by Muddam Sai Manish 
“Helping local stores go digital — one photo at a time.”
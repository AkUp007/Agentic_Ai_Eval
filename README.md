# **Agentic AI Evaluation Framework**

## **📌 Overview**
The Agentic AI Evaluation Framework is an automated system for the multi-dimensional evaluation of AI agent responses using a powerful **LLM-as-a-Judge** methodology. It provides a systematic, reproducible, and scalable solution to score AI agent outputs across multiple quality dimensions.
The included Streamlit application allows users to upload datasets, run evaluations, and visualize results interactively.

---

## **✨ Features**  

- **Multi-dimensional Scoring**: Evaluates responses across four core criteria: Instruction Following, Hallucination, Assumption Control, and Coherence/Accuracy.
- **Flexible Judge Models**: Supports multiple LLM providers, including Google Gemini, Groq, and models accessible via OpenRouter.
- **Parallel Processing**: Uses ThreadPoolExecutor to process large datasets asynchronously.
- **Interactive Dashboard**: Streamlit UI for uploading data, configuring evaluation parameters, and visualizing results.
- **Persistent & Resumable State**: Results are saved to CSV, enabling stop/resume functionality.
- **Simple CSV Workflow**: Input and output via CSV files for easy integration.

---

## **🛠 Tech Stack**  
| **Component** | **Technology** |  
|--------------|--------------|  
| **Web Framework** | Streamlit |  
| **Backend Logic** | Python 3.9+ |  
| **Data Handling** | Pandas |  
| **Judge Models** | Google Gemini, Groq, OpenRouter APIs | 
| **Visualization** | Matplotlib, Seaborn |  
| **Concurrency** | ThreadPoolExecutor | 

---

## **⚖️ Evaluation Framework**

This project leverages the LLM-as-a-Judge methodology. A capable language model is prompted with a detailed rubric to act as an impartial evaluator.

### **🔹 Evaluation Dimensions**

Each response is scored across four dimensions. Scores: 1 (Good), 0 (Neutral/Partial), -1 (Poor).

| **Dimension**         | **Description**                                        | **1 (Good)**               | **0 (Neutral)**             | **-1 (Poor)**                      |
| --------------------- | ------------------------------------------------------ | -------------------------- | --------------------------- | ---------------------------------- |
| Instruction Following | Adherence to all explicit constraints and instructions | Followed all instructions  | Minor deviation             | Ignored/violated instructions      |
| Hallucination         | Factually grounded and free of fabricated info         | No fabricated claims       | Minor speculation           | Contains major false claims        |
| Assumption Control    | Avoids or qualifies unjustified assumptions            | No unjustified assumptions | Minor, harmless assumptions | Major unjustified assumptions      |
| Coherence & Accuracy  | Clear, logical, and factually correct response         | Clear, logical, accurate   | Mild confusion/redundancy   | Confusing, disorganized, incorrect |

### **🔹 Scoring Formula**

The **Total Score** is the arithmetic mean of the four dimension scores:

**Total Score = (Instruction Following + Hallucination + Assumption Control + Coherence & Accuracy) / 4**

**Score range**: -1.0 to 1.0.

---

## **🚀 Getting Started**

### **🔹 Prerequisites**
Ensure you have:
- **Python 3.9+**
- **API keys for at least one supported provider:**
  - Google AI Studio (Gemini)
  - GroqCloud
  - OpenRouter.ai

### **📥 Installation & Configuration**

#### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/AkUp007/Agentic_Ai_Eval.git
cd Agentic_Ai_Eval
```

#### **2️⃣ Create a Virtual Environment (Optional but Recommended)**  
```bash
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### **3️⃣ Install Dependencies** 
```bash
pip install -r requirements.txt
```

#### **4️⃣ Set up API Keys**
Create a .env file in the project root:
```bash
# .env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
```
Note: Provide only the key(s) for the judge model(s) you plan to use.

---

## **⚙️ Usage**

### **▶️ Running the Application**
```bash
streamlit run stream.py
```
Access in your browser:
👉 **http://localhost:8501**

---

## **🔹 Workflow**

- **Upload CSV**: Must contain prompt and response columns.
- **Select Judge Model**: Choose from Gemini, Groq, or OpenRouter.
- **Configure Parameters**:
    - **Batch Size**: Rows per parallel batch
    - **Row Limit**: Total rows to evaluate (0 = all)
- **Start Evaluation**: Click "Start Evaluation"; results saved to results.csv.
- **Analyze & Visualize**: Use UI buttons to generate heatmaps and histograms.

---

## **🤖 Supported Judge Models**

| **Provider**  | **Access Method / Default Model**                            |
| ------------- | ------------------------------------------------------------ |
| Google Gemini | `google-generativeai` library, `gemini-2.5-flash`            |
| Groq          | `requests` library, `Llama 3.1 8B`                           |
| OpenRouter    | API access to Mistral AI, Meta, Google; default `Mistral 7B` |

---

## **📂 Project Structure**
```graphql
agentic-ai-evaluation/
├── stream.py              # Main Streamlit UI
├── eval1.py               # Core evaluation logic and scoring
├── prompt.py              # Contains JUDGE_RUBRIC_PROMPT
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── data/
    ├── input.csv          # Sample/input dataset
    └── results.csv        # Evaluation outputs
```
---

## **🔮 Roadmap**

- Add support for more evaluation dimensions (e.g., creativity, safety, bias)
- Implement detailed analytics dashboard for multi-run comparison
- Support comparison of multiple agent models side-by-side
- Package evaluator as an installable library

See open issues for full list of features and known issues.

---

## **🤝 Contributing**

Contributions are welcome!

- Fork the Project
- Create a Feature Branch (`git checkout -b feature/AmazingFeature`)
- Commit Changes (`git commit -m 'Add some AmazingFeature'`)
- Push to Branch (`git push origin feature/AmazingFeature`)
- Open a Pull Request

---

## **📜 License**

This project is licensed under the `MIT License`.

---

## **🙌 Acknowledgments**

Inspired by the need for scalable, objective AI evaluation
Built with Streamlit
Based on the LLM-as-a-Judge methodology

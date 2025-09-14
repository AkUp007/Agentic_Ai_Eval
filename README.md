🧠 Agentic AI Evaluation Framework
An automated framework for the multi-dimensional evaluation of AI agent responses using a powerful LLM-as-a-Judge methodology.

This project provides a systematic and scalable solution to score AI agent outputs against key quality dimensions, eliminating the need for slow and subjective manual grading. It features a user-friendly web interface for running evaluations and visualizing results.

📋 Table of Contents
About The Project

✨ Key Features

⚖️ The Evaluation Framework

Evaluation Dimensions

Scoring Formula

🚀 Getting Started

Prerequisites

Installation & Configuration

⚙️ Usage

Running the Application

Workflow

🤖 Supported Judge Models

📂 Project Structure

🔮 Roadmap

🤝 Contributing

📜 License

🙌 Acknowledgments

About The Project
As language models become more capable, evaluating the quality of their responses becomes increasingly complex. Manual evaluation is time-consuming, expensive, and often inconsistent. This framework addresses that challenge by automating the evaluation process.

Using a designated "Judge" LLM, it scores agent responses on a configurable set of criteria, providing structured, reproducible, and actionable feedback. The included Streamlit application makes it easy to upload datasets, run the evaluation pipeline, and analyze the results through interactive visualizations.

✨ Key Features
🔍 Multi-dimensional Scoring: Evaluates responses across four core criteria: Instruction Following, Hallucination, Assumption Control, and Coherence/Accuracy.

🤖 Flexible Judge Models: Supports multiple LLM providers for the judge, including Google Gemini, Groq, and any model on OpenRouter.

⚡ Parallel Processing: Utilizes a ThreadPoolExecutor to run evaluations asynchronously, dramatically speeding up the processing of large datasets.

📊 Interactive Dashboard: A Streamlit-powered UI to upload data, control evaluation parameters, and visualize results with heatmaps and histograms.

🔄 Persistent & Resumable State: The evaluation state is saved to a CSV file, allowing you to stop and resume processing without losing progress.

📁 Simple CSV Workflow: Uses CSV files for both input and output, making it easy to integrate with existing data pipelines.

⚖️ The Evaluation Framework
This project is built on the LLM-as-a-Judge methodology, where a capable language model is prompted with a detailed rubric to act as an impartial evaluator.

Evaluation Dimensions
Each agent response is scored across the following four dimensions. The score for each dimension is an integer: 1 (Good), 0 (Neutral/Partial), or -1 (Poor).

Dimension

Description

1 (Good)

0 (Neutral)

-1 (Poor)

Instruction Following

Does the agent adhere to all explicit constraints and instructions?

Followed all instructions

Minor deviation

Ignored/violated instructions

Hallucination

Is the response factually grounded and free of fabricated information?

No fabricated claims

Minor speculation or unverified claim

Contains major false claims

Assumption Control

Does the agent avoid or properly qualify unjustified assumptions?

No unjustified assumptions

Minor, harmless assumptions

Major unjustified assumptions

Coherence & Accuracy

Is the response clear, logically structured, and factually correct?

Clear, logical, and accurate

Mild confusion or redundancy

Confusing, disorganized, or incorrect

Scoring Formula
The Total Score is calculated as the arithmetic mean of the four dimension scores, providing a single metric for overall quality. The result is a float ranging from -1.0 to 1.0.

The formula is:


Total Score= 
4
∑ 
i=1
4
​
 score 
i
​
 
​
 
🚀 Getting Started
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.9+

API keys from at least one of the supported providers:

Google AI Studio (for Gemini)

GroqCloud

OpenRouter.ai

Installation & Configuration
Clone the repository:

git clone [https://github.com/your-username/agentic-ai-evaluation.git](https://github.com/your-username/agentic-ai-evaluation.git)
cd agentic-ai-evaluation

Install the dependencies:

pip install -r requirements.txt

Set up your API keys:
Create a file named .env in the root of the project directory. Copy the contents of .env.example (if provided) or add your keys in the following format:

# .env
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"

Note: You only need to provide the key for the judge model(s) you intend to use.

⚙️ Usage
Running the Application
Launch the Streamlit web interface with the following command:

streamlit run app.py

Open your web browser and navigate to http://localhost:8501.

Workflow
Upload CSV: Upload your input CSV file. It must contain prompt and response columns.

Select Judge Model: Choose the LLM you want to use for evaluation (e.g., Gemini, Groq).

Configure Parameters:

Batch Size: The number of rows to process in each parallel batch.

Row Limit: The total number of rows to evaluate from the input file (0 for all).

Start Evaluation: Click the "Start Evaluation" button. The progress will be logged to the console, and the UI will update upon completion. Results are automatically saved to results.csv.

Analyze & Visualize: Once the evaluation is complete, use the buttons in the UI to generate:

A heatmap showing the correlation between different evaluation dimensions.

Histograms displaying the distribution of the total score and individual dimension scores.

🤖 Supported Judge Models
The framework is configured to work with the following API providers out-of-the-box:

Google Gemini: via the google-generativeai library (uses gemini-1.5-flash).

Groq: via the requests library (uses Llama 3.1 8B by default).

OpenRouter: Can be used to access a wide variety of models, including those from Mistral AI, Meta, and Google (uses Mistral 7B by default).

You can easily switch between these models in the Streamlit UI.

📂 Project Structure
📦 agentic-ai-evaluation/
├── app.py                 # Main Streamlit application UI
├── evaluator.py           # Core evaluation logic and scoring functions
├── prompt.py              # Contains the JUDGE_RUBRIC_PROMPT
├── requirements.txt       # Project dependencies
├── README.md              # This documentation file
└── data/
    ├── input.csv          # Sample input dataset (or your uploaded data)
    └── results.csv        # Persisted evaluation scores and outputs

🔮 Roadmap
[ ] Add support for more evaluation dimensions (e.g., creativity, safety, bias).

[ ] Implement a detailed analytics dashboard for multi-run comparison.

[ ] Add support for comparing outputs from multiple agent models side-by-side.

[ ] Package the evaluator as an installable library.

See the open issues for a full list of proposed features (and known issues).

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.

🙌 Acknowledgments
Inspired by the growing need for scalable and objective AI evaluation.

Built with the awesome Streamlit framework.

Methodology based on the "LLM-as-a-Judge" concept.

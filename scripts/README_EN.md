# ArXiv AI/Algorithm Papers Daily Update

An automated system for fetching, analyzing, and organizing the latest AI/Algorithm research papers from ArXiv with AI-powered classification capabilities.

## Features

- **Automated Paper Retrieval**: Automatically fetches the latest AI/Algorithm papers from ArXiv
- **AI-Powered Analysis**: Uses Doubao LLM for intelligent paper categorization and analysis
- **Bilingual Support**: Provides paper titles in both English and Chinese
- **Code Link Detection**: Automatically extracts GitHub repository links
- **Organized Output**: Generates well-structured Markdown reports
- **Parallel Processing**: Utilizes multi-threading for improved efficiency
- **Smart Categorization**: Classifies papers into specific research areas
- **Core Contribution Extraction**: Uses LLM to extract core contributions from papers

## Project Structure

```
ArXiv_AI_Papers_Daily/
├── scripts/           # Script files
│   ├── get_ai_papers.py       # Main program
│   ├── ai_categories_config.py # AI/Algorithm classification config
│   ├── llm_helper.py          # LLM API helper
│   ├── doubao_client.py       # Doubao API client
│   ├── config.py              # Configuration file
│   └── requirements.txt      # Dependencies
├── data/              # Table-format paper information
│   └── YYYY-MM/
│       └── YYYY-MM-DD.md
└── local/             # Detailed paper information
    └── YYYY-MM/
        └── YYYY-MM-DD.md
```

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`
- Doubao API key
- Stable internet connection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure Doubao API key (in `config.py`)

## Configuration

Key parameters in `get_ai_papers.py`:
- `QUERY_DAYS_AGO`: Days to look back (0=today, 1=yesterday)
- `MAX_RESULTS`: Maximum number of papers to retrieve
- `MAX_WORKERS`: Maximum parallel processing threads

ArXiv Categories:
- cs.AI (Artificial Intelligence)
- cs.CL (Computation and Language)
- cs.CV (Computer Vision)
- cs.LG (Machine Learning)
- cs.MA (Multiagent Systems)
- cs.RO (Robotics)
- cs.MM (Multimedia)
- stat.ML (Statistics - Machine Learning)
- And other related categories

## Usage

Run the main script:
```bash
python get_ai_papers.py
```

### Output Files

The script generates two types of Markdown files:

1. Table Format (`data/YYYY-MM/YYYY-MM-DD.md`):
   - Basic paper information
   - Categorized by research areas
   - Concise tabular format

2. Detailed Format (`local/YYYY-MM/YYYY-MM-DD.md`):
   - Comprehensive paper details
   - AI-generated analysis
   - Core contributions
   - Code links

## Research Categories

Current supported research areas:

### AI/Algorithm Main Categories:
1. **AI Agents & Reasoning**
   - Single Agent Planning & Tool Use
   - Multi-Agent Collaboration
   - Long-Chain Reasoning & CoT
   - Context Engineering
   - Agent Evaluation & Benchmarks
   - Agentic Workflow & Automation

2. **LLM Architecture & Optimization**
   - Model Architecture Innovation
   - Efficient Attention Mechanisms
   - Parameter-Efficient Fine-tuning
   - Model Compression & Quantization
   - Inference Acceleration
   - Model Evaluation & Benchmarks

3. **Multimodal & Vision-Language**
   - Vision-Language Models
   - Multimodal Alignment
   - Image/Video Understanding
   - Cross-Modal Generation
   - Multimodal Reasoning

4. **Training & Optimization**
   - Training Strategies
   - Optimization Algorithms
   - Data Augmentation
   - Curriculum Learning
   - Distributed Training

5. **Evaluation & Benchmarks**
   - Model Evaluation Methods
   - New Benchmark Datasets
   - Performance Analysis
   - Robustness Evaluation

6. **Data & Knowledge**
   - Dataset Construction
   - Knowledge Graphs
   - Data Quality
   - Knowledge Enhancement

### Computer Vision (CV) Categories:
1. Visual Representation & Foundation Models
2. Visual Recognition & Understanding
3. Generative Visual Modeling
4. 3D Vision & Geometric Reasoning
5. Temporal Visual Analysis
6. Self-supervised & Representation Learning
7. Computational Efficiency & Model Optimization
8. Robustness & Reliability
9. Medical & Biological Imaging
10. Vision-Language & Multimodal

## Automated Deployment

Set up daily automatic runs using crontab:
```bash
# Edit crontab
crontab -e

# Add this line to run at 9 AM daily
0 9 * * * cd /path/to/ArXiv_AI_Papers_Daily/scripts && python get_ai_papers.py
```

## Error Handling

The script includes:
- ArXiv API rate limit handling
- Network error recovery
- Parallel processing error handling
- File system error handling
- LLM API call error handling

## Security Notes

- Store API keys securely
- Monitor API usage
- Keep dependencies updated
- Use virtual environments

## Technical Details

### LLM Classification
- Uses Doubao LLM for paper classification
- Supports two-level classification (main category and subcategory)
- Intelligent classification based on paper title and abstract
- Returns classification confidence scores

### Text Preprocessing
- Uses NLTK for text preprocessing
- Supports stemming and lemmatization
- Stop word filtering
- Tokenization

### Parallel Processing
- Uses ThreadPoolExecutor for parallel processing
- Configurable maximum thread count
- Progress bar display (using tqdm)

## Future Improvements

- Enhanced error recovery
- More research categories
- Better code link detection
- Interactive web interface
- Multi-language support
- Advanced paper filtering
- Support for more ArXiv categories
- Improved LLM classification accuracy

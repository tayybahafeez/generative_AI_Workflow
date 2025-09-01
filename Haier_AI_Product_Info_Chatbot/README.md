# Chatbot for Price Prediction

## Overview
A Retrieval-Augmented Generation (RAG) system for handling price-specific product queries using Gemini API and LlamaIndex.

## Features
- Preprocess Excel data into structured documents
- Query product prices using natural language
- Error handling for unavailable or ambiguous data

## Setup
1. **Python Version**: 
   - Python 3.9.19 recommended

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Key**: Add your Google API key to `.env`:
   ```
   Google_API_KEY=your_google_api_key

## Usage
**Run Application**:
```bash
python app.py
```

## Query Examples
- "What is the price of FFD Safety Device HW200-BC412345 5 Burner Hob?"
- "Tell me the cost of HW100-BP14929S3."

## Example Responses
- **Exact Match**: "The price of FFD Safety Device 5 Burner Hob is 77000."
- **No Price**: "Price not available for the specified product."
- **Multiple Matches**: Lists matching products with prices.

## Future Enhancements
- Support for more query types
- Add a web interface

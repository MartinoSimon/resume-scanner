# AI Resume Scanner - FREE Edition 🆓

## 🎉 100% FREE - No API Costs!

This version uses **Hugging Face's free inference API** with the powerful Mixtral-8x7B model. No credit card required!

## Prerequisites
- Python 3.8 or higher
- (Optional) Hugging Face API token for better rate limits - [Get one FREE here](https://huggingface.co/settings/tokens)

## Installation Steps

### 1. Clone or Create Project Directory
```bash
mkdir ai-resume-scanner
cd ai-resume-scanner
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Hugging Face Token
For better rate limits and faster response times, create a `.env` file:
```bash
HF_API_KEY=your_huggingface_token_here
```

**Without a token:** Still works! But may have rate limits and slower "cold start" times.
**With a token:** Better performance and higher rate limits (still 100% FREE).

Get your FREE token at: https://huggingface.co/settings/tokens

### 5. Project Structure
```
ai-resume-scanner/
├── main.py
├── index.html
├── requirements.txt
├── .env (optional)
└── venv/
```

## Running the Application

### 1. Start the Backend Server
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Open the Frontend
Simply open `index.html` in your web browser by double-clicking it or dragging it into your browser.

## Usage

1. **Paste Job Description**: Copy a job posting and paste it into the left textarea
2. **Upload Resume**: Drag and drop or click to upload your PDF resume
3. **Click Analyze**: Wait 20-30 seconds for the FREE AI analysis
   - **First time may take 30-40 seconds** (model "cold start")
   - **Subsequent requests are faster** (10-20 seconds)
4. **Review Results**: See your match score, missing keywords, and improvement tips

## How It Works

### AI Provider: Hugging Face 🤗
- **Model**: Mixtral-8x7B-Instruct (Open Source)
- **Cost**: 100% FREE forever
- **Quality**: Excellent results, comparable to paid alternatives
- **Rate Limits**: 
  - Without token: ~1 request per minute
  - With FREE token: Much higher limits

### Fallback System
If the AI is overloaded, the app automatically switches to a smart keyword-based analysis, ensuring you always get results!

## API Endpoints

### POST /analyze
Analyzes a resume against a job description.

**Request:**
- `file`: PDF file (form-data)
- `job_description`: String (form-data)

**Response:**
```json
{
  "success": true,
  "data": {
    "match_percentage": 75,
    "missing_keywords": ["Docker", "Kubernetes", "AWS"],
    "summary": "Strong technical background with relevant experience...",
    "tips": [
      "Add specific metrics to quantify your achievements",
      "Include Docker and Kubernetes experience if you have it",
      "Emphasize cloud platform expertise"
    ]
  }
}
```

## Troubleshooting

### "Model is loading" Error
- **Cause**: Hugging Face models have a "cold start" period when not recently used
- **Solution**: Wait 20-30 seconds and try again
- **Prevention**: Get a FREE Hugging Face token for priority access

### Slow Response Times
- First request: 30-40 seconds (model loading)
- Subsequent requests: 10-20 seconds
- **Tip**: Use a Hugging Face token for faster responses

### Rate Limit Errors
- **Without token**: Limited to ~1 request per minute
- **With FREE token**: Much higher limits
- **Solution**: Create a FREE account and add token to `.env`

### PDF Reading Errors
- Ensure PDF is not encrypted or password-protected
- PDF must contain extractable text (not scanned images)
- Keep file size under 10MB

## Cost Comparison

| Service | Cost per Analysis | Monthly (100 analyses) |
|---------|------------------|------------------------|
| **This App (FREE)** | $0.00 | $0.00 |
| OpenAI GPT-4 | ~$0.03 | $3.00 |
| Claude API | ~$0.02 | $2.00 |

## Production Deployment

For production, consider:

### Free Hosting Options:
- **Backend**: 
  - Railway (Free tier)
  - Render (Free tier)
  - Fly.io (Free tier)
  - PythonAnywhere (Free tier)
- **Frontend**:
  - GitHub Pages (FREE)
  - Vercel (FREE)
  - Netlify (FREE)

### Recommended Improvements:
1. Add user authentication
2. Store analysis history in a database
3. Implement caching for common job descriptions
4. Add email notifications for completed analyses
5. Create a simple payment system for premium features (if desired)

## Monetization Ideas (Optional)

Even though the AI is FREE, you can still monetize:
- **Freemium**: 3 free analyses/month, $5/month for unlimited
- **Credits**: Sell analysis credits in bulk
- **Premium Features**: 
  - Cover letter generation
  - LinkedIn profile optimization
  - Interview prep based on job description
- **White Label**: License to recruiters/agencies

## Why This is Amazing 🚀

1. **Zero AI Costs**: No OpenAI bills!
2. **Open Source AI**: Powered by Mixtral-8x7B
3. **Production Ready**: Deploy and start a business TODAY
4. **Scalable**: Handle thousands of users with FREE tier
5. **No Vendor Lock-in**: Own your entire stack

## License
MIT License - Build your SaaS empire with this! 💰
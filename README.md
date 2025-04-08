# YouTube Summary Bot

This project automatically processes daily videos from a specified YouTube channel (@parkerrexdaily by default) to generate summaries, extract vocabulary, and save the output.

## Features

- **Monitors** a YouTube channel for new uploads (using its RSS feed).
- **Downloads** the video (optional).
- Fetches the **transcript** (using `youtube-transcript-api`).
- **Summarizes** the content using an AI model via OpenRouter (configurable model).
- **Extracts** key vocabulary with definitions and example usage using NLP (spaCy) and AI (OpenRouter).
- **Generates** an output document (.docx format currently supported).
- **Uploads** the document to Dropbox.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd youtube_summary_bot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Configure API Keys and Settings:**
    - Create a `.env` file in the root directory.
    - Add your API keys to the `.env` file:
      ```dotenv
      OPENROUTER_API_KEY="your_openrouter_api_key" # Required for summary & definitions
      DROPBOX_ACCESS_TOKEN="your_dropbox_access_token" # Required for upload
      # YOUTUBE_API_KEY="your_youtube_api_key" # Optional, if using official API for detection later
      ```
    - Review and adjust settings in `config.json`. **Ensure `YOUTUBE_CHANNEL_ID` is correct for the channel you want to monitor.** You may also want to change the `OPENROUTER_MODEL` or `POLLING_INTERVAL_HOURS`.

## Running the Bot

```bash
python main.py
```

The script will run immediately to check the channel's RSS feed for the latest video. If a new video is found, it will be processed. The script will then continue running, checking the feed periodically based on the `POLLING_INTERVAL_HOURS` setting in `config.json`.

Press `Ctrl+C` to stop the scheduler.

## Folder Structure

```
/youtube_summary_bot/
├── .env               # Stores API keys (DO NOT COMMIT)
├── .gitignore         # Specifies intentionally untracked files
├── config.json        # Configuration settings (channel ID, model, etc.)
├── main.py            # Main application script
├── requirements.txt   # Python dependencies
├── downloads/         # Stores downloaded videos (if enabled)
├── logs/              # Stores log files and last processed video ID
│   ├── app.log
│   └── last_processed_id.txt
├── outputs/           # Stores generated summary documents
└── README.md          # This file
```

## TODO / Improvements

- Implement YouTube Data API v3 integration as an alternative, potentially more reliable, video detection method.
- Add support for Markdown (.md) output format in `generate_document`.
- Enhance vocabulary extraction logic (e.g., filter better, configurable POS tags, context-aware filtering).
- Improve error handling for OpenRouter API (rate limits, specific model errors, fallback models?).
- Add optional email/Slack notifications.
- Consider alternative transcript sources if `youtube-transcript-api` fails.
- Build a simple web interface for viewing summaries. 
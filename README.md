# Claude Code Experiments

This repository contains various coding experiments and projects created with the assistance of Claude AI. Each branch represents a different experiment or project, showcasing different technologies, frameworks, and use cases.

## Branches

### Claude Experiments

#### `claude/basketball-shot-chart-BvXWV`

A Python package for generating NBA-style hexbin shot charts, similar to those popularized by Kirk Goldsberry. Features include:

- Hexbin visualization of shot data
- Support for NBA API data fetching or synthetic demo data
- Customizable gridsize, color modes, and minimum shots per hex
- Output to PNG files

**Tech Stack:** Python, Matplotlib, NumPy, Pandas, NBA API (optional)

#### `claude/expense-tracker-app-H7R8o`

A modern, responsive Progressive Web App for personal finance tracking. Built as a PWA with offline-first architecture.

- Manual transaction entry and receipt scanning with OCR
- Smart category detection and multi-currency support
- Analytics dashboard with charts and spending predictions
- Budget setting with progress tracking
- Export/import functionality
- Dark mode and PWA installation support

**Tech Stack:** React, TypeScript, Vite, Tailwind CSS, Tesseract.js, Recharts, Workbox

#### `claude/file-conversion-cli-DczKJ`

A fast, zero-configuration CLI tool for converting files between various formats. Supports images, documents, data files, and videos.

- Single file and batch directory conversions
- Image formats: PNG, JPG, WebP, BMP, GIF, TIFF, ICO, HEIC/HEIF
- Document formats: Markdown to HTML/PDF/DocX, HTML to PDF
- Data formats: CSV, JSON, YAML, XLSX, TSV conversions
- Video formats: MP4, AVI, MKV, MOV, WebM, FLV, WMV, M4V, TS
- Quality and resize options

**Tech Stack:** Python, Click, Pillow, Pandoc, OpenPyXL, PyYAML, ffmpeg

#### `claude/flight-price-monitor-b4vhp`

A comprehensive flight price monitoring system focused on routes from Basque Country airports (BIO, EAS, VIT) to European and transatlantic destinations.

- Multi-provider support: Kiwi Tequila API, Amadeus, Google Flights via SerpAPI
- Smart deal detection comparing current prices to historical averages
- Flexible date searches and cabin bag filtering
- Watch mode for specific routes with price drop alerts
- Web dashboard with price trend charts
- Telegram notifications and email reports
- SQLite database for historical price tracking

**Tech Stack:** Python, Flask, Chart.js, SQLite, Telegram Bot API, SMTP

#### `claude/holiday-blog-2pgUJ`

A full-stack personal travel blog application with AI-powered insights and optional Office 365 integration.

- Travel documentation with photo galleries and interactive maps
- AI analysis for restaurants, recommendations, and highlights extraction
- Search and filtering by destination, favorites, timeline view
- OneNote and OneDrive integration for importing content
- JWT-based admin authentication
- Responsive design with Docker deployment options

**Tech Stack:** Node.js, Express, React, TypeScript, MongoDB, OpenAI API, Microsoft Graph API, Leaflet, Docker

#### `claude/personal-website-design-utqZY`

A modern, responsive personal website for Stefano Masneri, Senior AI Engineer. Features include:

- Interactive neural network visualization with Three.js
- Sections for about, research, publications, skills, blog, and contact
- Responsive design with smooth scrolling navigation
- Auto-geocoding and globe visualization
- Test suite for functionality

**Tech Stack:** HTML, CSS, JavaScript, Three.js, Leaflet

#### `claude/serie-a-predictor-nZVj4`

A football match prediction engine for Serie A, Premier League, La Liga, and Bundesliga using statistical modeling.

- Poisson distribution model with Dixon-Coles correction
- Expected Goals (xG) data integration from Understat
- Live bookmaker odds integration via The Odds API
- Value bet detection using expected value calculations
- Kelly Criterion stake sizing
- Multi-league support with team form and head-to-head statistics
- Web interface with best bets panel

**Tech Stack:** Python, Flask, HTML, CSS, JavaScript, football-data.org API, The Odds API, Understat

### Codex Experiments

#### `codex/create-image-classifications-branch-and-scripts`

Image classification and face recognition pipelines using machine learning.

- Image classification into categories: food, landscapes, people
- Multi-label face recognition (identifying multiple people in one image)
- Training scripts for custom classifiers
- Zero-shot classification using CLIP
- FastAPI web API for inference
- Face embedding enrollment and prediction

**Tech Stack:** Python, FastAPI, scikit-learn, CLIP, face_recognition, Uvicorn

## Getting Started

Each experiment is contained in its own branch. To explore a specific project:

1. Clone the repository:

   ```bash
   git clone https://github.com/Stocastico/Claude-code-experiments.git
   cd Claude-code-experiments
   ```

2. Switch to the desired branch:

   ```bash
   git checkout <branch-name>
   ```

3. Follow the setup instructions in the branch's README.md file.

## Contributing

This repository serves as a collection of experimental projects. Each branch is self-contained and may have its own contribution guidelines.

## License

See LICENSE file for details.</content>
<parameter name="filePath">/Users/fasteno/Documents/CODE/Claude-code-experiments/README.md

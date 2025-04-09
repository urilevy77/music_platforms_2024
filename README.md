# Music Streaming Analytics Dashboard

![Dashboard Preview](https://via.placeholder.com/800x400?text=Music+Streaming+Dashboard)

## Overview

This interactive Streamlit dashboard analyzes music streaming trends across various platforms. The application visualizes top artists, streaming patterns, platform correlations, and popularity trends using comprehensive music streaming data.

## Features

The dashboard is organized into four main tabs:

### 1. Overview
- Display total songs and unique artists in the dataset
- Show top songs by Spotify streams
- Visualize song releases per year
- Highlight top 10 artists by track count with their most popular songs

### 2. Trends and Events in Music Platforms
- Track streaming performance across YouTube, Spotify, TikTok and other platforms
- Visualize key industry events with annotations
- Compare platform growth with percentage change indicators
- Analyze streaming platform trends from 2015 to 2024

### 3. Platform Correlation
- Explore correlations between different streaming platforms with an interactive heatmap
- Generate scatter plots comparing metrics from different platforms
- Filter by top songs across selected platforms
- Normalize data to handle different scales

### 4. Artist Popularity Trend
- Compare the popularity trajectories of selected artists
- Analyze track release patterns over time
- View detailed artist performance metrics and best songs

## Data

The dashboard uses a comprehensive dataset (`data_project_final_2.csv`) containing information about:
- Songs and their artists
- Release dates
- Streaming counts across platforms (Spotify, YouTube, TikTok, etc.)
- Platform-specific metrics
- Track popularity scores

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-streaming-dashboard.git
cd music-streaming-dashboard
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Dependencies

- streamlit
- pandas
- plotly
- numpy
- scikit-learn
- pathlib

## Project Structure

```
music-streaming-dashboard/
│
├── app.py                  # Main Streamlit application
├── data/
│   └── data_project_final_2.csv    # Dataset
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

## Usage

1. Navigate to the project directory
2. Run the application with `streamlit run app.py`
3. The dashboard will open in your default web browser
4. Use the tabs to explore different analytics views
5. Interact with filters and selectors to customize the visualizations

## Features in Detail

### Data Processing
- The application handles various file encodings
- Transforms and normalizes data for visualization
- Processes timestamps for time-series analysis
- Scales metrics to handle large value disparities

### Visualizations
- Interactive line charts for trend analysis
- Heatmaps for correlation analysis
- Bar charts for count comparisons
- Scatter plots for relationship exploration
- Annotated timeline events

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sourced from music streaming platforms
- Built with Streamlit and Plotly

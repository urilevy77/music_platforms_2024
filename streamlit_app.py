import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
# Set the page configuration
st.set_page_config(
    page_title='The Battle of Metrics',
    page_icon=':musical_note:'
)
# Function to format large numbers
def format_number(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)
@st.cache_data
def load_data_schiffen():
    try:
        encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv('data/data_project_final_2.csv',
                    encoding=encoding,
                    nrows=4500,
                    on_bad_lines='skip'
                )
                df['Release.Date'] = pd.to_datetime(df['Release.Date'], errors='coerce')
                df['Year'] = df['Release.Date'].dt.year
                return df
            except UnicodeDecodeError:
                continue
        raise Exception("Could not read file with any of the attempted encodings")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def load_music_data():
    DATA_FILENAME = Path(__file__).parent / 'data/data_project_final_2.csv'
    df = pd.read_csv(DATA_FILENAME, encoding='ISO-8859-1')
    return df
def preprocessing_bassan(df):
    df['Release.Date'] = pd.to_datetime(df['Release.Date'])
    df.set_index('Release.Date', inplace=True)

    # Convert all numeric columns to numeric format, replacing -1 with 0
    numeric_cols = df.select_dtypes(include=['number', 'object']).columns
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '')
            .str.replace('-1', '0')
            .str.extract(r'(-?\d+\.?\d*)')[0]
            .fillna('0')
            .astype(float)
        )

    # Aggregate other platforms into a single column
    other_platforms = ['AirPlay.Spins', 'SiriusXM.Spins', 'Pandora.Streams', 'Deezer.Playlist.Reach', 'Soundcloud.Streams']
    df['other'] = df[other_platforms].sum(axis=1)

    return df

def preprocess_uri(df):
    exclude_cols = ['Track', 'Album.Name', 'Artist','Release.Date']

    # Identify numeric columns excluding the specified columns
    numeric_cols = [col for col in df.select_dtypes(include=['number', 'object']).columns 
                    if col not in exclude_cols]

    # Convert numeric columns
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '')
            .str.extract(r'(-?\d+\.?\d*)')[0]
            .fillna('0')
            .astype(float)
        )
    return df
important_platforms = [
        'TikTok.Views', 'Spotify.Streams', 'YouTube.Views', 
        'Shazam.Counts', 'Pandora.Streams'
    ]


st.title("Most streamed songs in 2024")

# Create tabs
tabs = st.tabs(["Overview", "Trends and events in music platforms", "Platform Correlation","Artist popularity trend"])
#####BASSAN####
# Tab 1: Overview
with tabs[0]:
    st.header("Overview")
  
    # Load data
    df_overview_table = load_music_data()
    df_overview_artists = load_music_data()
    preprocess_uri(df_overview_table)
    df_overview_table = df_overview_table[~(df_overview_table[important_platforms] == -1).any(axis=1)]
    df_overview_years=load_music_data()
    preprocessing_bassan(df_overview_years)
    df_sorted = df_overview_table.sort_values(by='Spotify.Streams', ascending=False)
    df_sorted = df_sorted.drop(columns=["Unnamed: 0"], errors="ignore")
    

    # Top Left: Songs Count
    col1, col2 = st.columns(2)
    with col1:
        total_songs = df_overview_artists.shape[0]
        st.subheader("Total Songs")
        st.write(total_songs)

    # Top Right: Unique Artists
    with col2:
        unique_artists = df_overview_artists['Artist'].nunique()
        st.subheader("Unique Artists")
        st.write(unique_artists)
        
    # Display the top songs
    st.write("Top Songs by Spotify Streams:")
   

    st.write(df_sorted.head())
    # Resample data for bar chart (number of songs released each year)
    df_monthly_counts = df_overview_years.resample('YE').size()

    # Create the bar chart using Plotly
    fig_bar = go.Figure()

    # Add bars for the monthly counts
    fig_bar.add_trace(go.Bar(
        x=df_monthly_counts.index,
        y=df_monthly_counts.values,
        marker_color='blue',
        name='Monthly Row Count',
        hovertemplate='<br> %{y} songs <extra></extra>'  # Tooltip for bar chart
    ))

    # Update bar chart layout
    fig_bar.update_layout(
        title='How Many Songs Released Each Year?',
        xaxis_title='Year',
        yaxis_title='Row Count',
        template='plotly_dark',
        hovermode='x unified',
        xaxis=dict(range=['1987-01-01', '2025-05-01'])
    )

    # Show bar chart
    st.plotly_chart(fig_bar)
   #### second visu #####
    
    
    
    # Count the number of tracks for each artist
    artist_counts = df_overview_artists['Artist'].value_counts().head(10)

    # Create a function to get the top song for each artist
    def get_top_song(artist):
        top_song = df_overview_artists[df_overview_artists['Artist'] == artist]['Track'].value_counts().idxmax()  # Get the most frequent song
        return top_song

    # Prepare the data for the bar chart
    artists = artist_counts.index
    track_counts = artist_counts.values
    top_songs = [get_top_song(artist) for artist in artists]

    # Define a color-blind friendly palette
    color_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#003F5C', '#FF7F0E', '#B95D61']

    # Create the bar chart using Plotly
    fig = go.Figure()

    # Add horizontal bars for the top 10 artists
    fig.add_trace(go.Bar(
        y=artists,
        x=track_counts,
        orientation='h',  # Horizontal bars
        marker_color=color_palette,
        name='Top 10 Artists by Track Count',
        hovertemplate=(
            '<b>Artist:</b> %{y}<br>' +
            '<b>Track Count:</b> %{x}<br>' +
            '<b>Top Song:</b> %{customdata}<br>'
            '<extra></extra>'
        ),
        customdata=top_songs  # Pass top songs directly
    ))

    # Add title and labels
    fig.update_layout(
        title='Top 10 Artists by Track Count',
        xaxis_title='Track Count',
        yaxis_title='Artist',
        xaxis=dict(range=[0, 70]),  # Set x-axis range
        template='plotly_dark',
        hovermode='y unified',  # Synchronize hover with the y-axis
        showlegend=False  # Hide legend
    )

    # Add the bar chart to the Streamlit dashboard
    st.plotly_chart(fig)
#### BASSAN ####
with tabs[1]:
    st.header("Trends in Music and Streaming Platforms")
    df_bass_first = load_music_data()
    df_bassan = preprocessing_bassan(df_bass_first)



    # Resample the data to monthly sums for platform analysis
    columns = ['YouTube.Views', 'Spotify.Streams', 'TikTok.Views', 'other']
    df_monthly = df_bassan.resample('MS').sum()

    # Apply a 6-month moving average to smooth the lines
    df_smoothed = df_monthly[columns].rolling(window=6, center=True).mean()

    # Create the line chart using Plotly
    fig_line = go.Figure()

    # Function to calculate percentage change for the last 6 months
    def calculate_percentage_change(platform, df):
        return ((df[platform].shift(0) - df[platform].shift(6)) / df[platform].shift(6)) * 100

    # Plot each platform's smoothed values as a line with percentage change in hover
    line_styles = ['solid', 'solid', 'solid', 'solid']
    colors = ['red', 'green', '#4A90A4', 'purple']
    for i, col in enumerate(columns):
        percentage_changes = calculate_percentage_change(col, df_smoothed)
        fig_line.add_trace(go.Scatter(
            x=df_smoothed.index,
            y=df_smoothed[col],
            mode='lines',
            name=col.replace('.', ' '),
            line=dict(width=2.5, color=colors[i % len(colors)], dash=line_styles[i % len(line_styles)]),
            hovertemplate='%{x}<br>Change: %{customdata:.2f}%<extra></extra>',  # Tooltip shows percentage change
            customdata=percentage_changes  # Add percentage change to hover
        ))

    events = {
    "[1]": ("2019-01-01", "TikTok enables product links in posts"),
    "[2]": ("2020-06-29", "India bans TikTok"),
    "[3]": ("2020-07-31", "Trump plans to ban TikTok"),
    "[4]": ("2021-02-10", "Biden halts Trump's TikTok ban"),
    "[5]": ("2018-05-01", "YouTube Music App is launched"),
    "[6]": ("2018-12-01", "Spotify Wrapped goes viral"),
    "[7]": ("2018-03-01", "Spotify goes public on NYSE"),
    }

    for event, (date, description) in events.items():
        event_date = pd.to_datetime(date)
        event_date = pd.Timestamp(date).replace(day=1)
        closest_date = df_smoothed.index[df_smoothed.index.to_period('M') == event_date.to_period('M')].min()

        # Determine which platform to annotate
        if "Spotify" in description:
            platform = 'Spotify.Streams'
        elif "YouTube" in description:
            platform = 'YouTube.Views'
        elif "TikTok" in description:
            platform = 'TikTok.Views'
        else:
            platform = 'other'

        # Check if the closest date exists in df_smoothed before accessing it
        if not closest_date is pd.NaT and closest_date in df_smoothed.index:
            fig_line.add_trace(go.Scatter(
                x=[closest_date],
                y=[df_smoothed.loc[closest_date, platform]],
                mode='markers',
                marker=dict(size=10, color='orange', line=dict(color='black', width=2)),
                hoverinfo='text',
                hovertext=f'{event}: {description}',
                showlegend=False  # Don't show in the legend
            ))

    # Update layout for the line chart
    fig_line.update_layout(
        title=dict(
            text='Streams Across Different Platforms (2015-2024)',
            y=0.9999
        ),
        xaxis_title='Year',
        yaxis_title='Streams and Views',
        template='plotly_dark',
        legend_title="Platforms",
        hovermode='x unified',
        xaxis=dict(range=['2014-01-01', '2024-05-01']),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5,
            traceorder='reversed',
            itemwidth=100,
        )
    )

    # Show the line chart
    st.plotly_chart(fig_line)

#### URI #########
with tabs[2]:
    st.header("Platform Correlation")
    
    df_uri=load_music_data()
    df_uri =preprocess_uri(df_uri)
    
    df_uri_sec=load_music_data()
    df_uri_numeric = preprocessing_bassan(df_uri_sec)


    
    # Remove rows with -1 in specified columns
    df_uri_cleaned = df_uri[~(df_uri[important_platforms] == -1).any(axis=1)]
   
    ### second ####
    # Platforms to choose from
           
    # Allow user to select metrics
    available_metrics = [
        'TikTok.Views',  'Spotify.Streams', 'YouTube.Views',
        'Shazam.Counts', 'Deezer.Playlist.Count', 'Spotify.Playlist.Count',
        'Pandora.Streams'
    ]
    selected_metrics = st.multiselect(
        "Select Metrics for Correlation",
        options=available_metrics,
        default=available_metrics
    )

    # Check if the required columns are selected
    if len(selected_metrics) > 1:
        # Calculate the correlation matrix
        correlation_matrix = df_uri_numeric[selected_metrics].corr().round(2)

        # Create the heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdYlBu',
            zmin=-1, zmax=1
        ))

        # Add annotations for the correlation values
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix.columns)):
                fig_heatmap.add_annotation(text=str(correlation_matrix.iloc[i, j]),
                                           x=correlation_matrix.columns[j],
                                           y=correlation_matrix.index[i],
                                           showarrow=False,
                                           font=dict(color="black"))

        fig_heatmap.update_layout(
            title='Correlation between Selected Metrics',
            xaxis_title='Metrics',
            yaxis_title='Metrics',
            template='plotly_dark'
        )

        # Show the heatmap
        st.plotly_chart(fig_heatmap)
    else:
        st.error("Please select at least two metrics for correlation")


    st.write("scatter diagram for correlation between platform")
    # Select platforms for sorting, x-axis, and y-axis
    sort_platform = st.selectbox("Select Platform for Top K Songs", important_platforms)
        # Slider to select top K songs
    k = st.slider('Select number of top songs',
                min_value=10,
                max_value=500,
                value=50,
                step=10)
    initial_x_selection = 'TikTok.Views'
    initial_y_selection = 'Spotify.Streams'

    # Create two columns for radio buttons
    col1, col2 = st.columns(2)

    with col1:
        x_platform = st.radio("Select Platform", important_platforms, index=important_platforms.index(initial_x_selection), horizontal=True)

    with col2:
        # Filter options for the second radio button to exclude the first selection
        y_platform_options = [platform for platform in important_platforms if platform != x_platform]
        y_platform = st.radio("Select Platform to Compare", y_platform_options, index=y_platform_options.index(initial_y_selection) if initial_y_selection in y_platform_options else 0, horizontal=True)

    # Check that both selections are unique and valid
    if x_platform == y_platform:
        st.warning("Please select two different platforms.")
    else:
        st.write(f"Selected platforms for comparison: {x_platform} and {y_platform}")
        # y_platform = st.selectbox("Select Platform to Compare", important_platforms, index=(important_platforms.index(sort_platform) + 1) % len(important_platforms))

    # Sort by selected platform and get top K songs
    df_top_k = df_uri_cleaned.nlargest(k, sort_platform)

    # Log transformation to handle skewed data
    df_log = df_top_k[important_platforms].apply(lambda x: np.log1p(x))

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply scaling
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_log),
        columns=important_platforms,
        index=df_top_k.index
    )

    # Create the scatter plot
    fig = go.Figure(
        go.Scatter(
            x=df_scaled[x_platform],
            y=df_scaled[y_platform],
            mode='markers',
            text=[f"Artist: {row['Artist']}<br>Track: {row['Track']}<br>" +
              f"{x_platform}: {format_number(row[x_platform])}<br>" +
              f"{y_platform}: {format_number(row[y_platform])}"
              for _, row in df_top_k.iterrows()],
        hovertemplate='<b>%{text}</b><extra></extra>'
        )
    )

    fig.update_layout(
        title=f'{x_platform} vs {y_platform} (Top {k} by {sort_platform})',
        xaxis_title=f'{x_platform.replace(".", " ")}',
        yaxis_title=f'{y_platform.replace(".", " ")}',
        template='plotly_dark'
    )

    # Show the scatter plot
    st.plotly_chart(fig)
    
  
        
        

    ### 3d visu ####
 
##### SCHIFFEN #######    
# Tab 3: Visualization
with tabs[3]:
   

    df = load_data_schiffen()
    # Updated list of artists
    ARTISTS = sorted([
        "Drake", "Taylor Swift", "Justin Bieber", "Miley Cyrus", 
        "Dua Lipa", "Billie Eilish", "Bad Bunny", "The Weeknd", 
        "Future", "Post Malone", "KAROL G", "Adele", "Coldplay",
        "Ariana Grande", "Bruno Mars", "Cardi B", "Charlie Puth"
    ])

    filtered_df = df[
        (df['Artist'].isin(ARTISTS)) & 
        (df['Year'] >= 2014) & 
        (df['Year'] <= 2023)
    ]

    yearly_stats = filtered_df.groupby(['Artist', 'Year']).agg({
        'Track.Score': 'mean',
        'Track': 'count'
    }).reset_index()

    best_songs = filtered_df.sort_values('Track.Score', ascending=False).groupby(['Artist', 'Year']).first().reset_index()

    st.header("Artist Performance Analysis")

    # Store year selection in session state
    if 'selected_years' not in st.session_state:
        st.session_state.selected_years = (2014, 2023)

    # Store selected artists in session state
    if 'selected_artists' not in st.session_state:
        st.session_state.selected_artists = ARTISTS[3:7]

    # Updated layout for artist selection - Add All button to the left
    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("Add All", use_container_width=True):
            st.session_state.selected_artists = ARTISTS

    with col2:
        selected_artists = st.multiselect(
            "Select Artists to Compare",
            ARTISTS,
            default=st.session_state.selected_artists,
            key='artist_selector',
            on_change=lambda: setattr(st.session_state, 'selected_artists', [])
        )
    # Create tabs
    tab1, tab2 = st.tabs(["Popularity Trends", "Release Patterns"])

    with tab1:
        # Create figure
        selected_years = st.slider(
            "Select Year Range",
            min_value=2014,
            max_value=2023,
            value=(2014, 2023),
            key="slider1"
        )
        
        fig1 = go.Figure()
        
        # Filter data based on selection and years
        plot_data = yearly_stats[
            (yearly_stats['Artist'].isin(selected_artists)) &
            (yearly_stats['Year'].between(selected_years[0], selected_years[1]))
        ]
        
        # Get best songs and scores for hover data
        yearly_best_songs = {}
        for year in range(2014, 2024):
            year_data = filtered_df[filtered_df['Year'] == year]
            scores = []
            for artist in selected_artists:
                artist_data = year_data[year_data['Artist'] == artist]
                if not artist_data.empty:
                    best_song = artist_data.loc[artist_data['Track.Score'].idxmax()]
                    scores.append({
                        'Artist': artist,
                        'Score': best_song['Track.Score'],
                        'Best Song': best_song['Track'],
                        'Average Score': artist_data['Track.Score'].mean(),
                        'Track Count': len(artist_data)
                    })
            yearly_best_songs[year] = sorted(scores, key=lambda x: x['Average Score'], reverse=True)

        if selected_artists:
            for artist in selected_artists:
                artist_data = plot_data[plot_data['Artist'] == artist]
                
                hover_text = []
                for year in artist_data['Year']:
                    artist_info = next((x for x in yearly_best_songs[year] if x['Artist'] == artist), None)
                    if artist_info:
                        hover_text.append(
                            f"<br>Artist: {artist}<br>" +
                            f"Average Score: {artist_info['Average Score']:.1f}<br>" +
                            f"Number of Tracks: {artist_info['Track Count']}<br>" +
                            f"Best Song: {artist_info['Best Song']}<br>"
                        )
                    
                fig1.add_trace(
                    go.Scatter(
                        x=artist_data['Year'],
                        y=artist_data['Track.Score'],
                        name=artist,
                        mode='lines+markers',
                        line=dict(shape='spline'),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_text
                    )
                )

        # Update layout with fixed axis ranges
        fig1.update_layout(
            title='Artist Popularity Trends Over Time',
            xaxis=dict(
                title='Year',
                type='linear',
                range=[selected_years[0], selected_years[1]],
                tickmode='linear',
                tick0=2014,
                dtick=1,
                constrain='domain'  # Constrain to specified range
            ),
            yaxis=dict(
                title='Average Track Score',
                gridcolor='LightGray',
                range=[0, 150]
            ),
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(b=20)
        )
        
        st.plotly_chart(fig1, use_container_width=True)

        

    with tab2:
        # Year range slider for tab2
        selected_years = st.slider(
            "Select Year Range",
            min_value=2014,
            max_value=2023,
            value=(2014, 2023),
            key="slider2"
        )
        
        # Filter based on selection and years for tab2
        plot_data_tab2 = yearly_stats[
            (yearly_stats['Artist'].isin(selected_artists)) &
            (yearly_stats['Year'].between(selected_years[0], selected_years[1]))
        ]
        
        fig2 = px.bar(
            plot_data_tab2,
            x='Year',
            y='Track',
            color='Artist',
            title='Number of Tracks Released per Year',
            labels={'Track': 'Number of Tracks', 'Year': 'Year'},
            barmode='stack'
        )
        
        fig2.update_layout(
            height=500,
            template='plotly_white',
            hovermode='x unified',
            xaxis=dict(
                range=[selected_years[0], selected_years[1]],
                tickmode='linear',
                tick0=2014,
                dtick=1,
                constrain='domain'
            ),
            margin=dict(b=20)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

import pandas as pd
import yt_dlp as youtube_dl

songs = pd.read_csv("data/audio.csv")
missing = songs[songs["video_url"] == ""]
print(f"Ratio: {missing / songs.shape[0]}")

def download_song(track_name, url):
    # Define youtube_dl options
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'mp3s/' + track_name + '.%(ext)s',  # Save the file as the title of the video
    }

    # Use youtube_dl to download the audio
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print(f"Downloaded: {url}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

for index, row in songs.iterrows():
    track_name = row['track_name']
    video_url = row['video_url']
    if video_url == "":
        continue
    download_song(track_name, video_url)

# Quick Redownload of Clean
# download_song("Clean", "https://www.youtube.com/watch?v=IGmMW7JTvuw&pp=ygUSdGF5bG9yIHN3aWZ0IGNsZWFu")
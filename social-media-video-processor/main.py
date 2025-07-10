#!/usr/bin/env python3
"""
Social Media Video Downloader & TV Effect Processor

This script downloads videos from different social media platforms (Instagram, TikTok),
adds a TV screen effect, overlays your logo, and prepares the video for YouTube Shorts posting.

Requirements:
- Python 3.8+
- Required packages: opencv-python, pillow, requests
- Optional packages (installed automatically if needed): pytube, instaloader, yt-dlp
- FFmpeg installed and in system PATH
"""

import os
import re
import argparse
import subprocess
import tempfile
import shutil
import sys
import time
from urllib.parse import urlparse
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import requests

# Define global paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(SCRIPT_DIR, "D:/youtube/downloads")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "D:/youtube/processed")
DEFAULT_LOGO_PATH = os.path.join(SCRIPT_DIR, "D:/youtube/logos and banners/logo.webp")
DEFAULT_TV_OVERLAY_PATH = os.path.join(SCRIPT_DIR, "D:/youtube/green screens/tvs.jpg")
DEFAULT_LINKS_FILE = os.path.join(SCRIPT_DIR, "D:/iCloudDrive/iCloud~is~workflow~my~workflows/links.txt")

# Store the FFmpeg path globally
FFMPEG_PATH = None

# YouTube Shorts recommended dimensions (9:16 aspect ratio)
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920

# Create necessary directories
os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process social media videos")
    parser.add_argument("--url", help="URL of the video to download (optional if --links-file is provided)")
    parser.add_argument("--links-file", help="Path to a text file containing URLs (one per line)",
                        default=DEFAULT_LINKS_FILE)
    parser.add_argument("--logo", help="Path to your logo image", default=DEFAULT_LOGO_PATH)
    parser.add_argument("--tv-effect", help="Apply TV screen effect", action="store_true")
    parser.add_argument("--tv-overlay", help="Path to TV overlay image", default=DEFAULT_TV_OVERLAY_PATH)
    parser.add_argument("--output-dir", help="Output directory for processed videos", default=PROCESSED_DIR)
    parser.add_argument("--shorts", help="Format video for YouTube Shorts (9:16 aspect ratio)", action="store_true",
                        default=True)
    parser.add_argument("--use-yt-dlp", help="Use yt-dlp for downloading all videos", action="store_true",
                        default=True)
    return parser.parse_args()


def check_ffmpeg():
    """Check if FFmpeg is installed and available in PATH."""
    try:
        # Check for FFmpeg in different possible locations
        ffmpeg_paths = [
            'ffmpeg',  # Default if in PATH
            'C:\\ffmpeg\\bin\\ffmpeg.exe',  # Common installation directory
            os.path.join(SCRIPT_DIR, 'ffmpeg.exe'),  # Local directory
        ]

        # Try each possible path
        for ffmpeg_path in ffmpeg_paths:
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.call([ffmpeg_path, '-version'], stdout=devnull, stderr=devnull)
                # If we got here, FFmpeg was found
                print(f"Found FFmpeg at: {ffmpeg_path}")
                # Set as global variable for other functions to use
                global FFMPEG_PATH
                FFMPEG_PATH = ffmpeg_path
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        # If we get here, FFmpeg wasn't found in any of the paths
        print("WARNING: FFmpeg not found in PATH. Audio may not be processed correctly.")
        print("Please install FFmpeg and make sure it's in your system PATH.")
        return False
    except Exception as e:
        print(f"Error checking for FFmpeg: {e}")
        return False


def identify_platform(url):
    """Identify the social media platform from the URL."""
    domain = urlparse(url).netloc

    if "instagram" in domain or "instagr.am" in domain:
        return "instagram"
    elif "tiktok" in domain:
        return "tiktok"
    elif "youtube" in domain or "youtu.be" in domain:
        return "youtube"
    else:
        raise ValueError(f"Unsupported platform: {domain}")


def download_with_yt_dlp(url, output_pattern):
    """Download a video using yt-dlp."""
    print(f"Downloading video using yt-dlp: {url}")

    # Make sure yt-dlp is installed
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error installing yt-dlp: {e}")

    # Generate a unique output filename
    timestamp = int(time.time())
    platform = identify_platform(url)
    output_path = os.path.join(DOWNLOADS_DIR, f"{platform}_{timestamp}.mp4")

    try:
        print(f"Running yt-dlp to download {url} to {output_path}")
        # Use --no-check-certificate to bypass SSL issues
        subprocess.check_call([
            "yt-dlp",
            "--no-check-certificate",
            "-o", output_path,
            url
        ])

        if os.path.exists(output_path):
            print(f"Download successful: {output_path}")
            return output_path
        else:
            print(f"Download failed: {output_path} not created")
            return None
    except Exception as e:
        print(f"Error downloading with yt-dlp: {e}")
        return None


def download_instagram_video(url):
    """Download a video from Instagram."""
    print(f"Downloading Instagram video: {url}")

    # Extract shortcode from URL
    shortcode_match = re.search(r'\/p\/([^\/\?]+)', url)
    if shortcode_match:
        shortcode = shortcode_match.group(1)
    else:
        # Use a timestamp if shortcode can't be extracted
        shortcode = f"instagram_{int(time.time())}"

    # Try direct download with yt-dlp first (most reliable method)
    output_path = os.path.join(DOWNLOADS_DIR, f"instagram_{shortcode}.mp4")
    print(f"Attempting to download with yt-dlp to {output_path}")

    try:
        result = download_with_yt_dlp(url, output_path)
        if result:
            return result
    except Exception as e:
        print(f"yt-dlp download failed: {e}")

    # Fallback to instaloader (only if yt-dlp fails)
    try:
        print("Trying instaloader as fallback...")
        # Install instaloader if not already installed
        try:
            import instaloader
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "instaloader"])
            import instaloader

        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            L = instaloader.Instaloader(dirname_pattern=temp_dir, download_videos=True)

            # Try to download as a post
            try:
                post = instaloader.Post.from_shortcode(L.context, shortcode)

                if post.is_video:
                    # Download the video
                    L.download_post(post, target=temp_dir)

                    # Find the video file
                    for file in os.listdir(temp_dir):
                        if file.endswith(".mp4"):
                            video_path = os.path.join(temp_dir, file)
                            shutil.copy2(video_path, output_path)
                            return output_path
            except Exception as post_error:
                print(f"Error downloading as post: {post_error}")
                # Continue with other methods

    except Exception as e:
        print(f"Instaloader method failed: {e}")

    # If all methods fail, raise an exception
    raise Exception("Failed to download Instagram video")


def download_tiktok_video(url):
    """Download a video from TikTok."""
    print(f"Downloading TikTok video: {url}")

    # Extract video ID from URL
    video_id = re.search(r'\/video\/(\d+)', url)
    if not video_id:
        path_parts = urlparse(url).path.split('/')
        video_id = next((part for part in path_parts if part.isdigit()), None)

    if not video_id:
        video_id = "tiktok_" + str(int(time.time()))
    else:
        video_id = video_id.group(1) if hasattr(video_id, 'group') else video_id

    output_path = os.path.join(DOWNLOADS_DIR, f"tiktok_{video_id}.mp4")

    # Use yt-dlp to download TikTok videos
    try:
        result = download_with_yt_dlp(url, output_path)
        if result:
            return result
    except Exception as e:
        print(f"Error downloading with yt-dlp: {e}")

    raise Exception("Failed to download TikTok video")


def download_youtube_video(url):
    """Download a video from YouTube."""
    print(f"Downloading YouTube video: {url}")

    # Use yt-dlp by default (most reliable)
    try:
        # Extract video ID
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        else:
            video_id = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
            video_id = video_id.group(1) if video_id else f"youtube_{int(time.time())}"

        output_path = os.path.join(DOWNLOADS_DIR, f"youtube_{video_id}.mp4")

        result = download_with_yt_dlp(url, output_path)
        if result:
            return result
    except Exception as e:
        print(f"yt-dlp download failed: {e}")

    # Fallback to pytube
    try:
        print("Trying pytube as fallback...")
        # Install pytube if not already installed
        try:
            from pytube import YouTube
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytube"])
            from pytube import YouTube

        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

        if not video:
            video = yt.streams.filter(file_extension="mp4").order_by("resolution").desc().first()

        if video:
            output_path = video.download(output_path=DOWNLOADS_DIR)
            return output_path
    except Exception as e:
        print(f"Pytube method failed: {e}")

    raise Exception("Failed to download YouTube video")


def download_video(url, use_yt_dlp=False):
    """Download a video from the supported platforms."""
    if use_yt_dlp:
        # Try using yt-dlp directly for any URL
        try:
            timestamp = int(time.time())
            output_path = os.path.join(DOWNLOADS_DIR, f"video_{timestamp}.mp4")
            result = download_with_yt_dlp(url, output_path)
            if result:
                return result
        except Exception as e:
            print(f"Direct yt-dlp download failed: {e}")
            # Continue with platform-specific methods

    # Platform-specific methods
    platform = identify_platform(url)

    if platform == "instagram":
        return download_instagram_video(url)
    elif platform == "tiktok":
        return download_tiktok_video(url)
    elif platform == "youtube":
        return download_youtube_video(url)


def format_for_shorts(video_path, output_path):
    """Format a video for YouTube Shorts (9:16 aspect ratio)."""
    print("Formatting video for YouTube Shorts...")

    # Normalize paths to use forward slashes
    video_path = os.path.normpath(video_path)
    output_path = os.path.normpath(output_path)

    # Make sure the input video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Make sure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Check if FFmpeg is available
    ffmpeg_available = FFMPEG_PATH is not None

    if ffmpeg_available:
        # Create a temporary file in the same directory as output
        temp_file = os.path.join(output_dir, f"temp_{os.path.basename(output_path)}")
        try:
            print(f"Using FFmpeg: {FFMPEG_PATH}")
            print(f"Input: {video_path}")
            print(f"Output: {temp_file}")

            # Format with FFmpeg for better quality (keeping audio)
            cmd = [
                FFMPEG_PATH, "-y",
                "-i", f'"{video_path}"',
                "-vf",
                f"scale=iw*max({SHORTS_WIDTH}/iw\,{SHORTS_HEIGHT}/ih):ih*max({SHORTS_WIDTH}/iw\,{SHORTS_HEIGHT}/ih),crop={SHORTS_WIDTH}:{SHORTS_HEIGHT}:(iw-{SHORTS_WIDTH})/2:(ih-{SHORTS_HEIGHT})/2",
                "-c:a", "copy",
                f'"{temp_file}"'
            ]

            # On Windows, we need to join the command as a string
            if sys.platform == "win32":
                cmd = " ".join(cmd)

            print(f"Running FFmpeg command: {cmd}")
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"FFmpeg stderr: {stderr.decode()}")
                raise Exception(f"FFmpeg failed with return code {process.returncode}")

            if os.path.exists(temp_file):
                # Remove existing output if it exists
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(temp_file, output_path)
                print(f"Successfully formatted video to {output_path}")
                return output_path
            else:
                raise FileNotFoundError(f"FFmpeg did not create output file: {temp_file}")

        except Exception as e:
            print(f"Error formatting video with FFmpeg: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            # Continue to OpenCV method as fallback

    # Rest of the OpenCV fallback code remains the same...
            # Continue to OpenCV method as fallback

    # Fallback to OpenCV if FFmpeg is not available or fails
    print("Using OpenCV to format video (no audio)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (SHORTS_WIDTH, SHORTS_HEIGHT))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get original dimensions
        h, w = frame.shape[:2]

        # Calculate scaling to maintain aspect ratio and fill the shorts format
        scale = max(SHORTS_WIDTH / w, SHORTS_HEIGHT / h)

        # Scale the frame
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))

        # Crop to exact shorts dimensions
        x_offset = (new_w - SHORTS_WIDTH) // 2
        y_offset = (new_h - SHORTS_HEIGHT) // 2

        # Make sure we have enough pixels to crop
        if new_w >= SHORTS_WIDTH and new_h >= SHORTS_HEIGHT:
            frame_cropped = frame_resized[y_offset:y_offset + SHORTS_HEIGHT, x_offset:x_offset + SHORTS_WIDTH]
        else:
            # Create black canvas and paste frame in center
            frame_cropped = np.zeros((SHORTS_HEIGHT, SHORTS_WIDTH, 3), dtype=np.uint8)
            paste_x = (SHORTS_WIDTH - new_w) // 2 if new_w < SHORTS_WIDTH else 0
            paste_y = (SHORTS_HEIGHT - new_h) // 2 if new_h < SHORTS_HEIGHT else 0
            paste_w = min(new_w, SHORTS_WIDTH)
            paste_h = min(new_h, SHORTS_HEIGHT)

            # Calculate source region (from resized frame)
            src_x = 0 if new_w <= SHORTS_WIDTH else (new_w - SHORTS_WIDTH) // 2
            src_y = 0 if new_h <= SHORTS_HEIGHT else (new_h - SHORTS_HEIGHT) // 2

            frame_cropped[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = frame_resized[src_y:src_y + paste_h,
                                                                                  src_x:src_x + paste_w]

        # Ensure dimensions are correct
        if frame_cropped.shape[:2] != (SHORTS_HEIGHT, SHORTS_WIDTH):
            frame_cropped = cv2.resize(frame_cropped, (SHORTS_WIDTH, SHORTS_HEIGHT))

        out.write(frame_cropped)

        processed_frames += 1
        if processed_frames % 30 == 0:
            print(f"Processed {processed_frames}/{frame_count} frames ({processed_frames / frame_count * 100:.1f}%)")

    cap.release()
    out.release()

    print(f"Successfully formatted video with OpenCV to {output_path}")
    return output_path


def add_tv_effect(video_path, tv_overlay_path, output_path):
    """Add a TV screen effect to the video."""
    print("Adding TV screen effect...")

    # Make sure the input video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Check if TV overlay exists, download a sample if not
    if not os.path.exists(tv_overlay_path):
        print(f"TV overlay not found at {tv_overlay_path}, creating a simple one...")
        # Create a simple TV overlay
        tv_overlay = np.zeros((SHORTS_HEIGHT, SHORTS_WIDTH, 4), dtype=np.uint8)
        # Draw a black border
        cv2.rectangle(tv_overlay, (0, 0), (SHORTS_WIDTH - 1, SHORTS_HEIGHT - 1), (0, 0, 0, 255), 50)
        # Make the center transparent
        tv_overlay[50:SHORTS_HEIGHT - 50, 50:SHORTS_WIDTH - 50, 3] = 0
        # Save the overlay
        cv2.imwrite(tv_overlay_path, tv_overlay)

    # Load the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load and resize the TV overlay
    tv_overlay = cv2.imread(tv_overlay_path, cv2.IMREAD_UNCHANGED)
    if tv_overlay is None:
        raise Exception(f"Could not load TV overlay: {tv_overlay_path}")

    # Check if the TV overlay has an alpha channel, if not add one
    if tv_overlay.shape[2] != 4:
        # Convert BGR to BGRA
        tv_overlay = cv2.cvtColor(tv_overlay, cv2.COLOR_BGR2BGRA)
        # Add alpha channel (fully opaque)
        tv_overlay[:, :, 3] = 255

    tv_overlay = cv2.resize(tv_overlay, (width, height))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to BGRA
        frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # Apply the TV overlay
        alpha_tv = tv_overlay[:, :, 3] / 255.0
        for c in range(3):
            frame_bgra[:, :, c] = frame_bgra[:, :, c] * (1 - alpha_tv) + tv_overlay[:, :, c] * alpha_tv

        # Convert back to BGR for writing
        frame_processed = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        out.write(frame_processed)

        processed_frames += 1
        if processed_frames % 30 == 0:
            print(f"Processed {processed_frames}/{frame_count} frames ({processed_frames / frame_count * 100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    # If FFmpeg is available, copy audio from original video
    if FFMPEG_PATH:
        # Copy audio from original to processed video
        temp_output = output_path + ".temp.mp4"
        try:
            os.rename(output_path, temp_output)

            # Use ffmpeg to copy audio
            ffmpeg_cmd = [
                FFMPEG_PATH, "-y",
                "-i", temp_output,
                "-i", video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as e:
            print(f"Warning: Could not copy audio: {e}")
            # Rename back the file if process failed
            if os.path.exists(temp_output) and not os.path.exists(output_path):
                os.rename(temp_output, output_path)
    else:
        print("WARNING: FFmpeg not found, audio will be lost in the processed video.")

    return output_path


def add_logo(video_path, logo_path, output_path):
    """Add a logo to the video."""
    print("Adding logo...")

    # Make sure the input video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Check if logo exists, create a sample if not
    if not os.path.exists(logo_path):
        print(f"Logo not found at {logo_path}, creating a sample...")
        # Create a simple logo
        logo = np.zeros((200, 200, 4), dtype=np.uint8)
        # Draw a white circle
        cv2.circle(logo, (100, 100), 80, (255, 255, 255, 255), -1)
        # Draw text
        cv2.putText(logo, "Logo", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0, 255), 2)
        # Save the logo
        cv2.imwrite(logo_path, logo)

    # Load the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load the logo
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise Exception(f"Could not load logo: {logo_path}")

    # Check if the logo has an alpha channel, if not add one
    if logo.shape[2] != 4:
        # Convert BGR to BGRA
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
        # Add alpha channel (fully opaque)
        logo[:, :, 3] = 255

    # Resize the logo (to about 15% of video height)
    logo_height = int(height * 0.15)
    logo_width = int(logo.shape[1] * logo_height / logo.shape[0])
    logo = cv2.resize(logo, (logo_width, logo_height))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Position for logo (bottom right corner with padding)
    x_pos = width - logo_width - 20
    y_pos = height - logo_height - 20

    # Process each frame
    processed_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make sure frame dimensions match video properties
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        # Create a copy of the frame to avoid modifying it directly
        frame_copy = frame.copy()

        # Calculate logo region boundaries
        end_y = min(y_pos + logo_height, height)
        end_x = min(x_pos + logo_width, width)

        # Adjust logo size if it doesn't fit
        logo_h = end_y - y_pos
        logo_w = end_x - x_pos

        if logo_h > 0 and logo_w > 0:
            # Resize logo if needed
            if logo_h != logo_height or logo_w != logo_width:
                logo_resized = cv2.resize(logo, (logo_w, logo_h))
            else:
                logo_resized = logo

            # Apply logo with alpha blending
            alpha_logo = logo_resized[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_logo

            for c in range(3):
                frame_copy[y_pos:end_y, x_pos:end_x, c] = (
                        alpha_frame * frame_copy[y_pos:end_y, x_pos:end_x, c] +
                        alpha_logo * logo_resized[:, :, c]
                )

        out.write(frame_copy)

        processed_frames += 1
        if processed_frames % 30 == 0:
            print(f"Processed {processed_frames}/{frame_count} frames ({processed_frames / frame_count * 100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    # If FFmpeg is available, copy audio from original video
    if FFMPEG_PATH:
        # Copy audio from original to processed video
        temp_output = output_path + ".temp.mp4"
        try:
            os.rename(output_path, temp_output)

            # Use ffmpeg to copy audio
            ffmpeg_cmd = [
                FFMPEG_PATH, "-y",
                "-i", temp_output,
                "-i", video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            # Remove temporary file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as e:
            print(f"Warning: Could not copy audio: {e}")
            # Rename back the file if process failed
            if os.path.exists(temp_output) and not os.path.exists(output_path):
                os.rename(temp_output, output_path)
    else:
        print("WARNING: FFmpeg not found, audio will be lost in the processed video.")

    return output_path


def get_processed_links_file(links_file_path):
    """Get the path for the processed links file and ensure it has a header."""
    links_dir = os.path.dirname(links_file_path)
    processed_file = os.path.join(links_dir, "processed_links.txt")

    # Create file with header if it doesn't exist
    if not os.path.exists(processed_file):
        with open(processed_file, 'w') as f:
            f.write("# Processed Links Archive\n")
            f.write("# Each batch is separated by a newline\n")
            f.write("# -------------------------\n\n")

    return processed_file


def read_all_processed_links(links_file_path):
    """Read all previously processed links from the archive."""
    processed_links_file = get_processed_links_file(links_file_path)
    processed_links = set()

    if os.path.exists(processed_links_file):
        try:
            with open(processed_links_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        processed_links.add(line)
        except Exception as e:
            print(f"Warning: Could not read processed links file: {e}")

    return processed_links


def archive_processed_links(links_file_path, urls):
    """Archive processed links and clean the original file, with newline separation between batches."""
    if not urls:
        return

    processed_links_file = get_processed_links_file(links_file_path)

    try:
        # Check if we need to add a newline before the new batch
        add_newline = os.path.exists(processed_links_file) and os.path.getsize(processed_links_file) > 0

        # Archive the processed links with newline separation
        with open(processed_links_file, 'a') as f:
            if add_newline:
                f.write(f"\n# Batch processed on {datetime.now()}\n")
            f.write("\n".join(urls) + "\n")

        print(f"Archived {len(urls)} links to: {processed_links_file}")

        # Clean the original file while preserving comments
        with open(links_file_path, 'r+') as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()

            for line in lines:
                line = line.strip()
                # Keep comments and empty lines
                if not line or line.startswith('#'):
                    f.write(line + "\n")
                # Remove lines that were processed
                elif line not in urls:
                    f.write(line + "\n")

        print(f"Cleaned original links file: {links_file_path}")

    except Exception as e:
        print(f"Error archiving links: {e}")


def read_links_from_file(file_path):
    """Read links from a text file, checking for previously processed ones."""
    if not os.path.exists(file_path):
        print(f"Links file not found: {file_path}")
        print("Creating an example links file...")

        with open(file_path, 'w') as f:
            f.write("# Add your links below (one per line)\n")
            f.write("# Examples:\n")
            f.write("# https://www.instagram.com/p/EXAMPLE/\n")
            f.write("# https://www.tiktok.com/@user/video/1234567890\n")

        print(f"Example links file created at: {file_path}")
        return []

    # Get already processed links
    processed_links = read_all_processed_links(file_path)
    new_links = []
    duplicate_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                if line in processed_links:
                    print(f"Skipping already processed link: {line}")
                    duplicate_count += 1
                else:
                    new_links.append(line)

    if duplicate_count:
        print(f"Found {duplicate_count} already processed links (skipped)")

    print(f"Found {len(new_links)} new links in {file_path}")
    return new_links


def get_next_video_number(output_dir, prefix="final_video_"):
    """
    Find the next available video number in the sequence.

    Args:
        output_dir: Directory to search for existing videos
        prefix: The filename prefix to look for

    Returns:
        The next available number in the sequence
    """
    max_num = 0
    if not os.path.exists(output_dir):
        return 1

    for filename in os.listdir(output_dir):
        if filename.startswith(prefix) and filename.endswith(".mp4"):
            # Extract number from filename (e.g., "final_video_42.mp4" -> 42)
            try:
                num = int(filename[len(prefix):-4])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    return max_num + 1


def cleanup_intermediate_files(*file_paths, keep_final=None, clear_links_file=False, links_file_path=None):
    """
    Delete intermediate processing files and optionally clear links file.

    Args:
        *file_paths: Variable list of file paths to potentially delete
        keep_final: The final file path to keep (won't be deleted)
        clear_links_file: Whether to clear the links file
        links_file_path: Path to the links file to clear
    """
    print("\nCleaning up intermediate files...")

    deleted_files = 0

    # Delete intermediate video files
    for file_path in file_paths:
        if file_path is None or file_path == keep_final:
            continue

        if not os.path.exists(file_path):
            continue

        try:
            os.remove(file_path)
            print(f"Deleted intermediate file: {file_path}")
            deleted_files += 1
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")

    # Clear links file if requested
    if clear_links_file and links_file_path:
        try:
            with open(links_file_path, 'w') as f:
                f.write("# Add new links here, one per line\n")
            print(f"Cleared links file: {links_file_path}")
        except Exception as e:
            print(f"Warning: Could not clear links file: {e}")

    print(f"Cleanup complete. Deleted {deleted_files} intermediate files.")
    return keep_final


def main():
    """Main function."""
    args = parse_arguments()
    check_ffmpeg()
    urls = []

    if args.url:
        urls.append(args.url)
    if args.links_file:
        file_urls = read_links_from_file(args.links_file)
        urls.extend(file_urls)

    if not urls:
        print("No URLs to process.")
        return

    # Get starting number for this batch
    start_num = get_next_video_number(args.output_dir)
    print(f"Starting video numbering from: {start_num}")

    # Track all files created during processing
    all_intermediate_files = []
    processed_urls = []

    for i, url in enumerate(urls):
        current_num = start_num + i
        print(f"\nProcessing URL {i + 1}/{len(urls)}: {url} (will be video #{current_num})")

        downloaded_path = None
        formatted_path = None
        tv_processed_path = None
        final_path = None

        try:
            # Download the video
            downloaded_path = download_video(url)
            print(f"Video downloaded to: {downloaded_path}")

            # Format for Shorts
            if args.shorts:
                video_basename = f"shorts_video_{current_num}.mp4"
                shorts_path = os.path.join(PROCESSED_DIR, video_basename)
                formatted_path = format_for_shorts(downloaded_path, shorts_path)
                current_path = formatted_path
            else:
                current_path = downloaded_path

            # TV Effect
            if args.tv_effect:
                video_basename = f"tv_video_{current_num}.mp4"
                tv_effect_path = os.path.join(PROCESSED_DIR, video_basename)
                tv_processed_path = add_tv_effect(current_path, args.tv_overlay, tv_effect_path)
                current_path = tv_processed_path

            # Add logo (final step)
            final_basename = f"final_video_{current_num}.mp4"
            final_path = os.path.join(args.output_dir, final_basename)
            final_path = add_logo(current_path, args.logo, final_path)

            print(f"Processing complete! Final video saved to: {final_path}")
            processed_urls.append(url)

            # Collect intermediate files for later cleanup
            if downloaded_path:
                all_intermediate_files.append(downloaded_path)
            if formatted_path:
                all_intermediate_files.append(formatted_path)
            if tv_processed_path:
                all_intermediate_files.append(tv_processed_path)

        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            import traceback
            traceback.print_exc()

            # Clean up any files created for this URL
            cleanup_intermediate_files(
                downloaded_path,
                formatted_path,
                tv_processed_path,
                keep_final=final_path
            )
            continue

    # After processing all URLs
    if processed_urls:
        print("\nProcessing complete. Performing final cleanup...")

        # Archive processed links if we used a links file
        if args.links_file and processed_urls:
            archive_processed_links(args.links_file, processed_urls)

        # Clean up intermediate files
        unique_intermediate_files = list(dict.fromkeys([f for f in all_intermediate_files if f is not None]))
        cleanup_intermediate_files(
            *unique_intermediate_files,
            keep_final=None  # We've already kept the final files
        )
    else:
        print("\nNo URLs were successfully processed.")


if __name__ == "__main__":
    main()
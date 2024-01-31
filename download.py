import json
import logging
import os
import subprocess
import sys
from itertools import product
from multiprocessing import Pool
from typing import List, Union

import pandas as pd
import typer
from pytube import YouTube


def run_command(cmd: List[str], verbose: bool, video_name: str) -> str:
    """
    Run a command.

    Args
    ----
        cmd: Command to run.
        verbose: Whether to show more information.

    Return
    ------
        The output of the command.
    """

    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )

    output = proc.stdout.strip()

    if verbose:
        logging.info("%s %s", video_name, output)

    return output


def trim_video(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
    **kwargs,
):
    """
    Trim a video file to extract a segment given its start and end.

    Args
    ----
        input_path (str): Path to video file.
        output_path (str): Path to output (where to write the output file).
        start (float): Time where the segment starts (in seconds).
        end (float): Time where the segment ends (in seconds).
        video_name (str): Name of the video clip.
        config (dict): Dict with general configurations for the pipeline.
    """
    video_name = kwargs.get("video_name", None)
    config = kwargs.get("config", None)

    try:
        if isinstance(output_path, str):
            run_command(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-ss",
                    f"{start}",
                    "-t",
                    f"{(end-start):.2f}",
                    "-loglevel",
                    config["video_processing"]["loglevel"],
                    "-y",
                    output_path,
                ],
                bool(config["video_processing"]["verbose"]),
                video_name,
            )
        else:
            run_command(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-ss",
                    f"{start}",
                    "-t",
                    f"{(end-start):.2f}",
                    "-loglevel",
                    config["video_processing"]["loglevel"],
                    "-y",
                    input_path,
                ],
                bool(config["video_processing"]["verbose"]),
                video_name,
            )

    except subprocess.CalledProcessError as e:
        print(
            e,
            e.stderr if e.stderr else "",
        )
        logging.info(e)


def extract_audio(
    input_path: str,
    output_path: str,
    start: float,
    end: float,
    **kwargs,
):
    """
    Extract audio from a video.

    Args
    ----
        input_path (str): Path to video clip.
        output_path (str): Path to output (where to write the output file).
        start (float): Time where the segment starts (in seconds).
        end (float): Time where the segment ends (in seconds).
        video_name (str): Name of the video clip.
        config (dict): Dict with general configurations for the pipeline.
    """

    video_name = kwargs.get("video_name", None)
    config = kwargs.get("config", None)

    try:
        if isinstance(output_path, str):
            run_command(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-f",
                    config["audio_processing"]["filetype"],
                    "-ab",
                    config["audio_processing"]["ab"],
                    "-acodec",
                    config["audio_processing"]["acodec"],
                    "-ac",
                    config["audio_processing"]["ac"],
                    "-ar",
                    config["audio_processing"]["ar"],
                    "-ss",
                    f"{start}",
                    "-t",
                    f"{(end-start):.2f}",
                    "-y",
                    "-loglevel",
                    config["audio_processing"]["loglevel"],
                    "-vn",
                    output_path,
                ],
                bool(config["audio_processing"]["verbose"]),
                video_name,
            )
        else:
            run_command(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-f",
                    config["audio_processing"]["filetype"],
                    "-ab",
                    config["audio_processing"]["ab"],
                    "-acodec",
                    config["audio_processing"]["acodec"],
                    "-ac",
                    config["audio_processing"]["ac"],
                    "-ar",
                    config["audio_processing"]["ar"],
                    "-ss",
                    f"{start}",
                    "-t",
                    f"{(end-start):.2f}",
                    "-y",
                    "-loglevel",
                    config["audio_processing"]["loglevel"],
                    "-vn",
                    input_path,
                ],
                bool(config["audio_processing"]["verbose"]),
                video_name,
            )

    except subprocess.CalledProcessError as e:
        logging.info(input_path)
        logging.info(e)


def download(video_data: list, config: dict):
    """
    Download a video from the Web.

    Args
    ----
        video_data (tuple): Tuple that contains information from the .CSV file.
            url (str): url for the video
            start (float): Time where the segment starts (in seconds).
            end (float): Time where the segment ends (in seconds).
            clip_name (str): Original video clip name.
            segment_name (str): Video segment name.
        config (dict): Dict with general configurations for the pipeline.

    Raises:
    -------
        NotImplementedError: Detection of video provider.
        In the meantime, we only support Youtube.
    """
    try:
        (url, start, end, clip_name, segment_name) = video_data
        # Create target folder if it does not exist
        video_path = os.fspath(os.path.join(config["output_dir"], "trimmed"))
        audio_path = os.fspath(os.path.join(config["output_dir"], "audios"))

        for _path in [config["output_dir"], video_path, audio_path]:
            os.makedirs(_path, exist_ok=True)

        # Placeholders for downloaded filename and path
        download_clip_filename = f"{clip_name}.{config['video_processing']['filetype']}"

        download_clip_filepath = os.path.join(
            config["output_dir"], download_clip_filename
        )
        ###

        if config["download"] and not os.path.exists(download_clip_filepath):
            if config["video_provider"] == "youtube":

                yt = YouTube(url)

                # Download video with the lowest resolution
                # that has both audio and video tracks
                yt_video = yt.streams.filter(
                    progressive=True,
                    file_extension=config["video_processing"]["filetype"],
                )
                yt_video = yt_video.get_lowest_resolution()

                assert yt_video is not None

                _download_clip_filepath = yt_video.download(
                    output_path=config["output_dir"]
                )

                os.rename(_download_clip_filepath, download_clip_filepath)

                # remove aux. vars.
                del _download_clip_filepath
            else:
                raise NotImplementedError(
                    "The only supported provider is Youtube."
                )

        # Preprocessing
        if config["preprocess"]:
            trim_video(
                download_clip_filepath,
                os.path.join(
                    video_path,
                    f"{clip_name}|{segment_name}.{config['video_processing']['filetype']}",
                ),
                start,
                end,
                video_name=f"{clip_name}|{segment_name}",
                config=config,
            )

            extract_audio(
                download_clip_filepath,
                os.path.join(
                    audio_path,
                    f"{clip_name}|{segment_name}.{config['audio_processing']['filetype']}",
                ),
                start,
                end,
                video_name=f"{clip_name}|{segment_name}",
                config=config,
            )

    except Exception as e:
        logging.info(e)
        #exit()


def main(
    csv_path: Union[None, str] = None,
    config_path: str = "params.json",
    cpu: Union[None, int] = 8,
    output_dir: str = "data/",
    skip_download: bool = True,
    preprocess: bool = False,
    multiprocessing: bool = False,
):
    """
    Download video content from a list of URLs provided into a .CSV file.

    Args:
        csv_path (str, optional): Path to .CSV that contains download asset information. Defaults to None. Each line in the .CSV should contain the following: <URL>,<START_SEC>,<END_SEC>,<FINAL_VIDEO_NAME>
        config_path (str, optional): Path to configuration file. Defaults to "params.json".
        cpu (Union[None, int], optional): Number of parallel downloading processes. Defaults to 8.
        output_dir (str, optional): Download output directory. Defaults to "data/".
        skip_download (bool, optional): Flag used to skip downloading. Defaults to False.
        preprocess (bool, optional): Flag to signal whether to preprocess the data using FFMPEG. Defaults to False.
        multiprocessing (bool, optional): Flag to signal whether to use multiprocessing. Defaults to False.
    """
    if csv_path is None:
        raise ValueError("You must provide a csv file.")

    # create logging file
    logging.basicConfig(
        filename=f"{os.path.splitext(csv_path)[0]}.log",
        filemode="w",
        format="%(message)s",
        level=logging.INFO,
    )

    # Load configs
    with open(config_path, encoding="utf-8") as f:
        config = json.loads(f.read())

    config["output_dir"] = output_dir
    config["preprocess"] = preprocess
    config["download"] = skip_download

    # read URLs
    data = pd.read_csv(csv_path, header=None)

    # download each file in different processes or in a single process
    if multiprocessing:
        with Pool(cpu) as pool:
            pool.starmap(
                download, product(data.to_records(index=False), [config])
            )
    else:
        for i, row in data.iterrows():
            sys.stdout.write(f"{i}/{data.shape[0]}\r")
            sys.stdout.flush()

            download(row.tolist(), config)


if __name__ == "__main__":
    typer.run(main)

# Video Downloader

## Installation

Use the `requirements.txt` file to install the required Python packages:

```bash
pip install -r requirements.txt
```

## Description

This Python script downloads video content from a list of URLs provided in a .CSV file. Additionally, it also preprocesses the downloaded data using FFMPEG.

The .CSV file should have the following structure:

```
<URL>,<START_SEC>,<END_SEC>,<FINAL_VIDEO_NAME>
...
```

`START_SEC` and `END_SEC` are used by FFMPEG to trim the original video and extract the audio track. To configure the behaviour of FFMPEG use the auxiliary `params.json`. `FINAL_VIDEO_NAME` will be used to rename the downloaded/generated files.

```bash
python download.py [OPTIONS] --csv-path=PATH_TO_CSV_FILE
```

### Options

```bash
-h, --help                          Print this help text and exit
--csv-path                          Path to .CSV that contains download asset
                                    information. Defaults to None.
--config-path                       Path to configuration file.
                                    Defaults to "params.json".
--cpu                               Number of parallel downloading processes.
                                    Defaults to 8.
--output                            Download output directory.
                                    Defaults to "data/".
--skip-download                     Flag used to skip the downloading part of
                                    the pipeline preprocess the. Defaults to False.
--preprocess                        Flag to signal whether to preprocess the
                                    data using FFMPEG. Defaults to False.
--multiprocessing                   Flag to signal whether to use multiprocessing.
                                    Defaults to False.
```

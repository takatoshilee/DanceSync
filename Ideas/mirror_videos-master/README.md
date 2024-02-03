# mirror_videos
A Python script that mirrors (flips left-right horizontally) videos given to it in an input directory using [OpenCV](https://opencv.org). Leaves much to be desired, but may be useful in gesture and sign language research if a large number of videos needs to be flipped horizontally (e.g., to artifically inverse the handedness of the person performing a gesture in a video).

<h1>Example</h1>

[Input video](example/gebaerdensprache.mp4)  |  [Mirrored output video](example/gebaerdensprache_mirror.mp4)
:-------------------------:|:-------------------------:
<img src="example/gebaerdensprache.png" width="95%" /> | <img src="example/gebaerdensprache_mirrored.png" width="95%" />

Example video of the German Sign Language (*Deutsche Gebärdensprache*, DGS) sign for "sign language" courtesy of [Henrike Maria Falke, gebaerdenlernen.de](http://www.gebaerdenlernen.de/index.php?article_id=88) ([Creative Commons by-nc-sa/3.0/de](https://creativecommons.org/licenses/by-nc-sa/3.0/de/)).

<h1>How-to</h1>

The script is `argparse`-enabled, so it can be used from the command line:

```
patrick$ python mirror_videos.py --help
usage: mirror_videos.py [-h] [-t] [-id INPUTDIR] [-od OUTPUTDIR] [-s SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  -t, --filetype        File type (e.g., 'mp4'; use without preceding '.').
  -id INPUTDIR, --inputdir INPUTDIR
                        Path where files that should be mirrored are located.
  -od OUTPUTDIR, --outputdir OUTPUTDIR
                        Path to directory to which output files should be
                        written.
  -s SUFFIX, --suffix SUFFIX
                        The given string will be added to the file name of
                        mirrored videos (e.g., given 'mirror' 'video1.mp4'
                        will become 'video1_mirror.mp4').
```

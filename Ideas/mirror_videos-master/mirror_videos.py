#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:29:21 2018

@author: Patrick C. Trettenbrein, trettenbrein@cbs.mpg.de
"""

import os
import cv2
import argparse

# Set up argument parser to use script from command line
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--filetype", help="File type (e.g., 'mp4'; use \
                    without preceding '.').", action="store_true")
parser.add_argument("-id", "--inputdir", help="Path where files that should\
                    be mirrored are located.", type=str)
parser.add_argument("-od", "--outputdir", help="Path to directory to which\
                    output files should be written. If the given directory\
                    doesn't exist, it can be created by the script.", type=str)
parser.add_argument("-s", "--suffix", help="The given string will be added\
                    to the file name of mirrored videos (e.g., given 'mirror'\
                    'video1.mp4' will become 'video1_mirror.mp4').", type=str)
args = parser.parse_args()

# Check whether a file type was passed to the program, if not default to MP4
if args.filetype:
    filetype = args.filetype
else:
    filetype = "mp4"

# Was an input path passed to the program? If not, default to script directory
if args.inputdir:
    if os.path.exists(args.inputdir):
        directory = args.inputdir
    else:
        print("Path does not exist. Defaulting to script directory.")
        directory = os.getcwd()

os.chdir(directory)

# Was an output path passed to the program? If not, default to input directoy
if args.outputdir:
    if os.path.exists(args.outputdir):
        output_directory = args.outputdir
    else:
        dir_q = input("Output directory does not exist:\n%s\n Do you want to\
                      create it? Type either 'y' or 'n'." % args.outputdir)
        dir_q_resp = dir_q[0].lower()

        while True:
            if dir_q_resp == "" or not dir_q_resp in ["y", "n"]:
                print("You can only type 'y' or 'n'.")
            else:
                break

        if dir_q_resp == "y":
            # Create given input directory
            os.makedirs(args.outputdir)
            output_directory = args.outputdir
        if dir_q_resp == "n":
            # Default to input directory
            output_directory = directory
            print("All mirrored files will be saved to the input directory.")
else:
    output_directory = directory

# Was a suffix for output files passed to the program? Defaults to "mirror"
if args.suffix:
    suffix = args.suffix
else:
    suffix = "mirror"

# Determine total number of files of given type in input directory
total_files = len([f for f in os.listdir(directory) if f.endswith(filetype)])

if total_files == 0:
    print("No '%s' files to process in the folder %s" % (filetype, directory))
else:
    print("A total of %s '%s' files will be processed.\n" % (total_files,
                                                             filetype))

# Iterate over files of given type in input directory
for c, filename in enumerate([f for f in os.listdir(directory) if
                              f.endswith(filetype)]):
    print("Processing file '%s' (%s of %s)." % (filename, c+1,
          total_files))
    video = cv2.VideoCapture(filename)

    # Gather info about input video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fn, ext = os.path.splitext(os.path.basename(filename))
    out = cv2.VideoWriter("%s/%s_%s%s" % (output_directory, fn, suffix, ext),
                          fourcc, fps, (width, height))

    # Flip video frame by frame and write to output file
    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            frame = cv2.flip(frame, 1)
            out.write(frame)
        else:
            break

    video.release()
    out.release()

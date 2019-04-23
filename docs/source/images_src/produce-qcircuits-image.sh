#!/bin/bash

pdflatex $1
pdfcrop $1.pdf
convert -density 300 -quality 90 $1-crop.pdf $1.png

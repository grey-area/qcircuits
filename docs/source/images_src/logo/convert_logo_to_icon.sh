#!/bin/bash

convert -density 384 -background transparent logo_square.svg -define icon:auto-resize -colors 256 icon.ico

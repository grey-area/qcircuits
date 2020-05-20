#!/bin/bash

# For the static site generated from this documentation to work with
# fastpages, we need to remove the underscore from directory names.

rm build/fastpages -rf

make html

cp -r build/html build/fastpages

cd build/fastpages

find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/_images/images/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/_sources/sources/g'
find . \( -type d -name .git -prune \) -o -type f -print0 | xargs -0 sed -i 's/_static/static/g'

mv _images images
mv _sources sources
mv _static static

cd ../..

printf "Loading ASVSpoof2019...\n"
axel -n 8 https://datashare.ed.ac.uk/download/DS_10283_3336.zip
unzip DS_10283_3336.zip
unzip LA.zip
mkdir data
mv LA data
mv *.pdf data
mv README.txt data
mv LICENSE_text.txt data
rm *.zip

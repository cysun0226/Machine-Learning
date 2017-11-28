unzip 0416045.zip &&
chmod +x 0416045/run.sh && 
cd 0416045/ &&
./run.sh ../train.csv ../test.csv > output.txt
cat output.txt

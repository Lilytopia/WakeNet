clear

cd ./overlaps
rm *.so
make
cd ../

cd ./nms
rm *.so
make
cd ../../

cd ./_cdht
rm *.so
make
cd ../../

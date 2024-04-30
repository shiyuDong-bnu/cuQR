#g++ ../src/serial.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o serial.o
#g++ ../src/block_s.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o block_s.o
#g++ ../src/block.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o block.o
#g++ ../test/gramma.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o gramma.o
#g++ ../src/tile.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o tile.o -g
g++ ../src/parallel.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o parallel.o -g -fopenmp
g++ ../src/compare.cpp -I$HOME/lib/armadillo/include -L$HOME/lib/armadillo/lib -larmadillo  -o compare.o -g -fopenmp

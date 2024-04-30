for i in 2 4 8 16 32 64
do
echo "$i "
./compare.o $i 45  45  4 
done

for i in {2..9}
do
  scp -r ~/@ddl hduser@hadoop${i}:~
done

for i in {2..9}
do
  ssh -t hduser@hadoop${i} "rm -r ~/@ddl"
done

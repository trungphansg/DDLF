for i in {2..9}
do
  ssh hduser@hadoop${i} "$1"
done
